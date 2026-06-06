import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
import time
import chromadb
from sentence_transformers import SentenceTransformer


# API
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

VECTORSTORE_DIR = "./vectorstore"
INSPECTIONS_DIR = "./data/inspections"
os.makedirs(INSPECTIONS_DIR, exist_ok=True)


# Inicialização
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
collection = chroma_client.get_or_create_collection(
    name="inspections",
    metadata={"hnsw:space": "cosine"}
)


# Sumário com LLM
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

Responde APENAS com o texto do summary, sem introdução, sem explicação, sem markdown.
"""

def generate_summary(inspection, max_retries=3):
    """Gera um summary textual rico para indexação."""
    inspection_text = json.dumps(inspection, ensure_ascii=False, indent=2)
    prompt = f"{PROMPT_SUMMARY}\n\nInspeção:\n{inspection_text}"

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0)
            )
            return response.text.strip()

        except ClientError as e:
            if e.code in [429, 503]:
                wait = 35 + attempt * 15
                print(f"  [aviso] Erro {e.code}, a aguardar {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Limite de tentativas excedido")


# Indexação
def index_inspection(inspection):
    """
    Indexa uma inspeção na vector store.
    Estratégia híbrida: summary como chunk principal + metadata estruturada.
    """
    inspection_id = inspection.get("inspection_id", str(uuid.uuid4()))

    # verifica se já está indexada
    existing = collection.get(ids=[inspection_id])
    if existing["ids"]:
        print(f"  [skip] {inspection_id} já indexada")
        return inspection_id

    # gera summary
    print(f"  A gerar summary para {inspection_id}...")
    summary = generate_summary(inspection)
    time.sleep(6)  # rate limiting

    # gera embedding
    embedding = embedding_model.encode(summary).tolist()

    # metadata estruturada para filtragem pre-retrieval
    timestamp = inspection.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(timestamp)
        hour = dt.hour
        weekday = dt.weekday()  # 0=segunda, 6=domingo
        date_str = dt.strftime("%Y-%m-%d")
    except:
        hour, weekday, date_str = 0, 0, ""

    issues = inspection.get("issues", [])
    issue_types = list(set(i.get("type", "") for i in issues))
    has_critical = any(i.get("severity") == "high" for i in issues)

    metadata = {
        "inspection_id": inspection_id,
        "zone_id": inspection.get("zone_id", ""),
        "overall_status": inspection.get("overall_status", ""),
        "fill_rate": float(inspection.get("shelf_fill_rate", 1.0)),
        "issue_count": len(issues),
        "issue_types": ",".join(issue_types),
        "has_critical": str(has_critical),
        "timestamp": timestamp,
        "date": date_str,
        "hour": hour,
        "weekday": weekday,
        "summary": summary
    }

    # indexa na vector store
    collection.add(
        ids=[inspection_id],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[metadata]
    )

    # guarda inspeção em disco
    path = os.path.join(INSPECTIONS_DIR, f"{inspection_id}.json")
    inspection["summary"] = summary
    with open(path, "w", encoding="utf-8") as f:
        json.dump(inspection, f, indent=2, ensure_ascii=False)

    print(f"  [ok] Indexada: {inspection_id}")
    print(f"  Summary: {summary[:100]}...")
    return inspection_id


def index_directory(inspections_dir=None):
    """Indexa todas as inspeções de uma pasta."""
    if inspections_dir is None:
        inspections_dir = INSPECTIONS_DIR

    files = list(Path(inspections_dir).glob("*.json"))
    print(f"A indexar {len(files)} inspeções...")

    for i, path in enumerate(files):
        print(f"[{i+1}/{len(files)}] {path.name}")
        with open(path, encoding="utf-8") as f:
            inspection = json.load(f)
        index_inspection(inspection)


# Retrieval
def retrieve(query, k=3, zone_filter=None):
    """
    Recupera as k inspeções mais relevantes para uma query.
    Opcionalmente filtra por zona.
    """
    query_embedding = embedding_model.encode(query).tolist()

    where = None
    if zone_filter:
        where = {"zone_id": {"$eq": zone_filter}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "inspection_id": results["ids"][0][i],
            "summary": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity": round(1 - results["distances"][0][i], 3)
        })

    return retrieved


# Query com síntese LLM
PROMPT_RAG_ANSWER = """És um assistente de análise de inspeções de prateleiras de supermercado.

Com base nas inspeções históricas recuperadas abaixo, responde à query do gestor.

Regras:
- Cita sempre as inspeções relevantes pelo inspection_id e data
- Se os dados não forem suficientes para responder, diz isso claramente
- Responde em português
- Sê direto e objetivo

Inspeções recuperadas:
{context}

Query: {query}
"""

def query_memory(query, k=3, zone_filter=None, max_retries=3):
    """
    Responde a uma query em linguagem natural usando o histórico de inspeções.
    """
    retrieved = retrieve(query, k=k, zone_filter=zone_filter)

    if not retrieved:
        return "Não foram encontradas inspeções relevantes no histórico.", []

    # constrói contexto
    context_parts = []
    for r in retrieved:
        meta = r["metadata"]
        context_parts.append(
            f"[{r['inspection_id']}] {meta.get('date', '')} {meta.get('hour', '')}h "
            f"— Zona: {meta.get('zone_id', '')} "
            f"— Status: {meta.get('overall_status', '')} "
            f"— Fill rate: {meta.get('fill_rate', '')} "
            f"— Issues: {meta.get('issue_types', 'nenhum')}\n"
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

        except ClientError as e:
            if e.code in [429, 503]:
                wait = 35 + attempt * 15
                print(f"  [aviso] Erro {e.code}, a aguardar {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Limite de tentativas excedido")


# Estatísticas
def get_stats():
    """Retorna estatísticas da vector store."""
    count = collection.count()
    print(f"Total de inspeções indexadas: {count}")
    return count


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso:")
        print("  python src/rag_memory.py index <inspection.json>")
        print("  python src/rag_memory.py index-dir [pasta]")
        print("  python src/rag_memory.py query \"<pergunta>\" [k] [zona]")
        print("  python src/rag_memory.py stats")
        sys.exit(1)

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