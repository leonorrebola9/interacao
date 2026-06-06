import os
import json
import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv


# API
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"
CACHE_DIR = "./cache/inspections"
os.makedirs(CACHE_DIR, exist_ok=True)

VALID_STATUSES = ["ok", "warning", "critical"]
VALID_SEVERITIES = ["low", "medium", "high"]
VALID_ISSUE_TYPES = ["empty_shelf", "wrong_product", "damaged", "misaligned", "label_missing", "other"]


# Prompts
PROMPT_A_ZERO_SHOT = """Analisa esta imagem de uma prateleira de supermercado e produz uma análise estruturada em JSON.

O JSON deve seguir exatamente este schema:
{
  "overall_status": "ok|warning|critical",
  "issues": [
    {
      "issue_id": "ISS_001",
      "type": "empty_shelf|wrong_product|damaged|misaligned|label_missing|other",
      "location": "descrição da localização na imagem",
      "severity": "low|medium|high",
      "description": "descrição em linguagem natural do problema",
      "confidence": 0.0,
      "affected_area_pct": 0.0
    }
  ],
  "shelf_fill_rate": 0.0,
  "products_detected": ["lista de categorias de produto visíveis"],
  "model_reasoning": "explica aqui o teu raciocínio antes de classificar"
}

Regras importantes:
- "affected_area_pct" é um valor entre 0.0 e 1.0 (ex: 0.35 significa 35% da área)
- "shelf_fill_rate" é um valor entre 0.0 e 1.0
- "confidence" é um valor entre 0.0 e 1.0
- Usa APENAS estes tipos de issue: empty_shelf, wrong_product, damaged, misaligned, label_missing, other
- Espaço no fim de uma prateleira não é empty_shelf — só classifica como tal se houver posições claramente sem produto onde deveria haver
- Se não houver problemas, "issues" deve ser lista vazia e "overall_status" deve ser "ok"

Responde APENAS com o JSON. Sem texto adicional, sem markdown, sem ```json.
"""

PROMPT_B_COT = """Analisa esta imagem de uma prateleira de supermercado seguindo estes passos de raciocínio obrigatórios:

PASSO 1 — DESCRIÇÃO GERAL
Descreve o que vês na imagem: quantas prateleiras, que tipo de produtos, contexto geral.

PASSO 2 — ANÁLISE ZONA A ZONA
Para cada prateleira visível (superior, meio, inferior), descreve:
- Está cheia, parcialmente vazia ou completamente vazia?
- Os produtos estão alinhados e bem posicionados?
- Há produtos tombados, danificados ou fora do lugar?
- As etiquetas de preço estão presentes e visíveis?

PASSO 3 — IDENTIFICAÇÃO DE ANOMALIAS
Lista todas as anomalias encontradas com localização exata e severidade estimada.
Nota: espaço no fim de uma prateleira não é empty_shelf. Só classifica como empty_shelf se houver posições claramente sem produto onde deveria haver.

PASSO 4 — CLASSIFICAÇÃO FINAL
Com base nos passos anteriores, produz o JSON final seguindo estas regras:
- "affected_area_pct" entre 0.0 e 1.0 (ex: 0.35 = 35% da área)
- "shelf_fill_rate" entre 0.0 e 1.0
- "confidence" entre 0.0 e 1.0
- Usa APENAS estes tipos: empty_shelf, wrong_product, damaged, misaligned, label_missing, other

{
  "overall_status": "ok|warning|critical",
  "issues": [
    {
      "issue_id": "ISS_001",
      "type": "empty_shelf|wrong_product|damaged|misaligned|label_missing|other",
      "location": "descrição da localização na imagem",
      "severity": "low|medium|high",
      "description": "descrição em linguagem natural do problema",
      "confidence": 0.0,
      "affected_area_pct": 0.0
    }
  ],
  "shelf_fill_rate": 0.0,
  "products_detected": ["lista de categorias de produto visíveis"],
  "model_reasoning": "resumo do raciocínio dos passos anteriores"
}

Escreve os 4 passos e termina com o JSON. Sem markdown, sem ```json à volta do JSON final.
"""

PROMPT_C_FEW_SHOT = """Analisa esta imagem de uma prateleira de supermercado.

Tipos de issue válidos (usa APENAS estes): empty_shelf, wrong_product, damaged, misaligned, label_missing, other
Valores numéricos: affected_area_pct, shelf_fill_rate e confidence são sempre entre 0.0 e 1.0

Aqui estão dois exemplos de análises anteriores corretas:

EXEMPLO 1 — Prateleira normal:
Imagem: prateleira de bebidas com garrafas de água alinhadas, etiquetas visíveis, sem espaços vazios.
JSON:
{
  "overall_status": "ok",
  "issues": [],
  "shelf_fill_rate": 0.95,
  "products_detected": ["água", "bebidas"],
  "model_reasoning": "Prateleira completamente preenchida, produtos alinhados, sem anomalias visíveis."
}

EXEMPLO 2 — Prateleira com problemas:
Imagem: prateleira de cereais com secção central vazia (4-5 posições) e uma caixa tombada no lado direito.
JSON:
{
  "overall_status": "critical",
  "issues": [
    {
      "issue_id": "ISS_001",
      "type": "empty_shelf",
      "location": "prateleira central, secção do meio",
      "severity": "high",
      "description": "4-5 posições consecutivas sem produto na secção central",
      "confidence": 0.9,
      "affected_area_pct": 0.35
    },
    {
      "issue_id": "ISS_002",
      "type": "misaligned",
      "location": "prateleira central, lado direito",
      "severity": "low",
      "description": "Caixa de cereais tombada",
      "confidence": 0.85,
      "affected_area_pct": 0.05
    }
  ],
  "shelf_fill_rate": 0.65,
  "products_detected": ["cereais"],
  "model_reasoning": "Secção central vazia e produto tombado identificados."
}

Agora analisa a imagem submetida e produz o JSON completo seguindo o mesmo padrão.
Responde APENAS com o JSON. Sem texto adicional, sem markdown, sem ```json.
"""


# Normalização da resposta
def clean_response(data):
    # overall_status
    status = data.get("overall_status", "warning").lower()
    data["overall_status"] = status if status in VALID_STATUSES else "warning"

    # issues
    cleaned_issues = []
    for i, issue in enumerate(data.get("issues", [])):
        issue_type = issue.get("type", "other").lower()
        severity = issue.get("severity", "low").lower()

        # normaliza tipo — mapeia variantes comuns para o schema correto
        type_map = {
            "misplaced_product": "wrong_product",
            "misplaced": "wrong_product",
            "out_of_place": "wrong_product",
            "empty": "empty_shelf",
            "damage": "damaged",
            "missing_label": "label_missing",
        }
        issue_type = type_map.get(issue_type, issue_type)
        if issue_type not in VALID_ISSUE_TYPES:
            issue_type = "other"
        if severity not in VALID_SEVERITIES:
            severity = "low"

        # normaliza valores numéricos
        confidence = float(issue.get("confidence", 0.8))
        if confidence > 1.0:
            confidence = confidence / 100.0

        affected = float(issue.get("affected_area_pct", 0.0))
        if affected > 1.0:
            affected = affected / 100.0

        cleaned_issues.append({
            "issue_id": issue.get("issue_id", f"ISS_{i+1:03d}"),
            "type": issue_type,
            "location": issue.get("location", ""),
            "severity": severity,
            "description": issue.get("description", ""),
            "confidence": round(confidence, 2),
            "affected_area_pct": round(affected, 2)
        })

    data["issues"] = cleaned_issues

    # shelf_fill_rate
    fill = float(data.get("shelf_fill_rate", 0.8))
    if fill > 1.0:
        fill = fill / 100.0
    data["shelf_fill_rate"] = round(fill, 2)

    return data


# Cache
def get_cache_path(image_path, strategy):
    md5 = hashlib.md5(open(image_path, "rb").read()).hexdigest()
    return os.path.join(CACHE_DIR, f"{md5}_{strategy}.json")

def load_from_cache(image_path, strategy):
    cache_path = get_cache_path(image_path, strategy)
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return None

def save_to_cache(image_path, strategy, data):
    cache_path = get_cache_path(image_path, strategy)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Parsing
def parse_inspection_json(raw_text, image_path, zone_id, strategy):
    import re
    matches = re.findall(r'\{[\s\S]*\}', raw_text)
    if not matches:
        raise ValueError("Nenhum JSON encontrado na resposta")

    data = json.loads(matches[-1])
    data = clean_response(data)

    data["inspection_id"] = f"INS_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6].upper()}"
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    data["image_path"] = str(image_path)
    data["zone_id"] = zone_id
    data["strategy"] = strategy

    return data


# Inspeção principal
def inspect_image(image_path, zone_id="Z_UNKNOWN", strategy="A", max_retries=3):
    image_path = str(image_path)

    cached = load_from_cache(image_path, strategy)
    if cached:
        print(f"  [cache] {Path(image_path).name} (estratégia {strategy})")
        return cached

    prompts = {"A": PROMPT_A_ZERO_SHOT, "B": PROMPT_B_COT, "C": PROMPT_C_FEW_SHOT}
    if strategy not in prompts:
        raise ValueError(f"Estratégia inválida: {strategy}. Usa 'A', 'B' ou 'C'.")

    with open(image_path, "rb") as f:
        image_data = f.read()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                    prompts[strategy]
                ],
                config=types.GenerateContentConfig(temperature=0)
            )
            raw_text = response.text.strip()
            result = parse_inspection_json(raw_text, image_path, zone_id, strategy)
            save_to_cache(image_path, strategy, result)
            return result

        except ClientError as e:
            if e.code in [429, 503]:
                wait = 35 + attempt * 15
                print(f"  [aviso] Erro {e.code}, a aguardar {wait}s...")
                time.sleep(wait)
            else:
                raise
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [aviso] Erro a parsear JSON (tentativa {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise

    raise RuntimeError("Limite de tentativas excedido")


def inspect_directory(images_dir, zone_id="Z_UNKNOWN", strategy="A", delay=6):
    images = list(Path(images_dir).glob("*.jpg"))
    print(f"A inspecionar {len(images)} imagens com estratégia {strategy}...")
    results = []
    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] {img_path.name}...", end=" ")
        try:
            result = inspect_image(str(img_path), zone_id=zone_id, strategy=strategy)
            print(f"{result.get('overall_status', '?')} | fill_rate: {result.get('shelf_fill_rate', '?')}")
            results.append(result)
            time.sleep(delay)
        except Exception as e:
            print(f"ERRO: {e}")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python shelf_inspector.py <imagem.jpg> [zone_id] [estrategia A|B|C]")
        print("Exemplo: python src/shelf_inspector.py data/test/img_001.jpg Z_S1 B")
        sys.exit(1)

    img = sys.argv[1]
    zone = sys.argv[2] if len(sys.argv) > 2 else "Z_S1"
    strat = sys.argv[3] if len(sys.argv) > 3 else "A"

    print(f"\nA inspecionar: {img}")
    print(f"Zona: {zone} | Estratégia: {strat}\n")

    result = inspect_image(img, zone_id=zone, strategy=strat)
    print(json.dumps(result, indent=2, ensure_ascii=False))