"""
rag_memory.py
Componente 3 — Memória histórica de inspeções com FAISS + embeddings Gemini
"""

import os
import json
import uuid
import numpy as np
import faiss
import pickle
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 3072

VECTORSTORE_DIR = "./vectorstore"
INSPECTIONS_DIR = "./data/inspections"
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss.index")
METADATA_PATH = os.path.join(VECTORSTORE_DIR, "metadata.pkl")

os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(INSPECTIONS_DIR, exist_ok=True)


# FAISS
def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        print(f"  [faiss] Índice carregado: {index.ntotal} entradas")
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        metadata = []
        print("  [faiss] Novo índice criado")
    return index, metadata

def save_index(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)


# Embeddings
def get_embedding(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text
            )
            vec = np.array(response.embeddings[0].values, dtype=np.float32)
            vec = vec / np.linalg.norm(vec)
            return vec

        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                wait = 35 + attempt * 15
                print(f"  [aviso] Servidor indisponível, a aguardar {wait}s (tentativa {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Limite de tentativas excedido no embedding")


# Sumário
PROMPT_SUMMARY = """És um sistema de indexação de inspeções de prateleiras de supermercado.

Dado o seguinte resultado de inspeção em JSON, gera um summary rico em termos semanticamente relevantes para recuperação futura.

O summary deve incluir obrigatoriamente (quando disponível):
- Zona inspecionada
- Data e hora
- Fill rate
- Tipos de produtos detetados
- Problemas encontrados com localização e severidade
- Estado geral

Exemplo de bom summary:
"prateleira inferior da zona Z_S3 com fill rate de 72%, produto de limpeza (detergente líquido) fora de posição na secção central, embalagem danificada detetada no lado direito, terça-feira 15h, estado warning"

Exemplo de mau summary:
"prateleira com problemas"

Responde APENAS com o texto do summary, sem introdução, sem explicação, sem markdown."""

def generate_summary(inspection, max_retries=3):
    # contexto de afluência
    zone_id = inspection.get("zone_id", "")
    timestamp = inspection.get("timestamp", "")
    
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from journey_context import get_affluence_context
        hour = datetime.fromisoformat(timestamp).hour
        affluence = get_affluence_context(zone_id, hour)
        print(f"  [afluência] {zone_id} às {hour}h: '{affluence}'")
    except Exception as e:
        print(f"  [afluência] erro: {e}")
        affluence = ""

    inspection_text = json.dumps(inspection, ensure_ascii=False, indent=2)
    affluence_note = f"\nContexto de afluência histórica: {affluence}" if affluence else ""
    prompt = f"{PROMPT_SUMMARY}\n\nInspeção:\n{inspection_text}{affluence_note}"

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0)
            )
            return response.text.strip()

        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                wait = 35 + attempt * 15
                print(f"  [aviso] Servidor indisponível, a aguardar {wait}s (tentativa {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Limite de tentativas excedido no summary")


# Indexação
def index_inspection(inspection):
    inspection_id = inspection.get("inspection_id", str(uuid.uuid4()))
    index, metadata = load_index()

    if any(m["inspection_id"] == inspection_id for m in metadata):
        print(f"  [skip] {inspection_id} já indexada")
        return inspection_id

    print(f"  A gerar summary para {inspection_id}")
    summary = generate_summary(inspection)
    print(f"  Summary: {summary[:100]}")
    time.sleep(6)

    print(f"  A gerar embedding")
    embedding = get_embedding(summary)

    timestamp = inspection.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(timestamp)
        hour = dt.hour
        weekday = dt.weekday()
        date_str = dt.strftime("%Y-%m-%d")
    except:
        hour, weekday, date_str = 0, 0, ""

    issues = inspection.get("issues", [])
    issue_types = list(set(i.get("type", "") for i in issues))
    has_critical = any(i.get("severity") == "high" for i in issues)

    meta = {
        "inspection_id": inspection_id,
        "zone_id": inspection.get("zone_id", ""),
        "overall_status": inspection.get("overall_status", ""),
        "fill_rate": float(inspection.get("shelf_fill_rate", 1.0)),
        "issue_count": len(issues),
        "issue_types": issue_types,
        "has_critical": has_critical,
        "timestamp": timestamp,
        "date": date_str,
        "hour": hour,
        "weekday": weekday,
        "summary": summary
    }

    index.add(embedding.reshape(1, -1))
    metadata.append(meta)
    save_index(index, metadata)

    path = os.path.join(INSPECTIONS_DIR, f"{inspection_id}.json")
    inspection["summary"] = summary
    with open(path, "w", encoding="utf-8") as f:
        json.dump(inspection, f, indent=2, ensure_ascii=False)

    print(f"  [ok] Indexada: {inspection_id} | Total: {index.ntotal}")
    return inspection_id


def index_directory(inspections_dir=None):
    if inspections_dir is None:
        inspections_dir = INSPECTIONS_DIR

    files = list(Path(inspections_dir).glob("*.json"))
    print(f"A indexar {len(files)} inspeções")

    for i, path in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {path.name}")
        with open(path, encoding="utf-8") as f:
            inspection = json.load(f)
        index_inspection(inspection)


# Retrieval
def retrieve(query, k=3, zone_filter=None):
    index, metadata = load_index()

    if index.ntotal == 0:
        return []

    query_embedding = get_embedding(query).reshape(1, -1)
    search_k = min(k * 3, index.ntotal)
    scores, indices = index.search(query_embedding, search_k)

    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]

        if zone_filter and meta.get("zone_id") != zone_filter:
            continue

        retrieved.append({
            "inspection_id": meta["inspection_id"],
            "summary": meta["summary"],
            "metadata": meta,
            "similarity": round(float(score), 3)
        })

        if len(retrieved) >= k:
            break

    return retrieved


# Query
PROMPT_RAG_ANSWER = """És um assistente de análise de inspeções de prateleiras de supermercado.

Com base nas inspeções históricas recuperadas abaixo, responde à query do gestor.

Regras:
- Cita sempre as inspeções relevantes pelo inspection_id e data
- Se os dados não forem suficientes para responder, diz isso claramente
- Responde em português
- Sê direto e objetivo

Inspeções recuperadas:
{context}

Query: {query}"""

def query_memory(query, k=3, zone_filter=None, max_retries=3):
    retrieved = retrieve(query, k=k, zone_filter=zone_filter)

    if not retrieved:
        return "Não foram encontradas inspeções relevantes no histórico.", []

    context_parts = []
    for r in retrieved:
        meta = r["metadata"]
        context_parts.append(
            f"[{r['inspection_id']}] {meta.get('date', '')} {meta.get('hour', '')}h "
            f"— Zona: {meta.get('zone_id', '')} "
            f"— Status: {meta.get('overall_status', '')} "
            f"— Fill rate: {meta.get('fill_rate', '')} "
            f"— Issues: {', '.join(meta.get('issue_types', [])) or 'nenhum'}\n"
            f"Summary: {r['summary']}"
        )

    context = "\n\n---\n\n".join(context_parts)
    prompt = PROMPT_RAG_ANSWER.format(context=context, query=query)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0)
            )
            return response.text.strip(), retrieved

        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                wait = 35 + attempt * 15
                print(f"  [aviso] Servidor indisponível, a aguardar {wait}s (tentativa {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Limite de tentativas excedido na query")


# Estatísticas
def get_stats():
    index, metadata = load_index()
    print(f"Total de inspeções indexadas: {index.ntotal}")
    return index.ntotal



if __name__ == "__main__":
    import sys

    command = sys.argv[1]

    if command == "index":
        if len(sys.argv) < 3:
            print("Erro: falta o ficheiro de inspeção.")
            sys.exit(1)
        with open(sys.argv[2], encoding="utf-8") as f:
            inspection = json.load(f)
        index_inspection(inspection)

    elif command == "index-dir":
        folder = sys.argv[2] if len(sys.argv) > 2 else INSPECTIONS_DIR
        index_directory(folder)

    elif command == "query":
        if len(sys.argv) < 3:
            print("Erro: falta a query.")
            sys.exit(1)
        q = sys.argv[2]
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        zona = sys.argv[4] if len(sys.argv) > 4 else None
        answer, sources = query_memory(q, k=k, zone_filter=zona)
        print(f"\nResposta:\n{answer}")
        print(f"\nFontes usadas ({len(sources)}):")
        for s in sources:
            print(f"  [{s['inspection_id']}] similaridade: {s['similarity']}")

    elif command == "stats":
        get_stats()

    else:
        print(f"Comando desconhecido: {command}")