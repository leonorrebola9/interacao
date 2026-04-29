import json
import argparse
import re
from pathlib import Path
from datetime import datetime, timezone

import requests

# constantes
ollama_url  = "http://localhost:11434/api/generate"
model       = "llama3.1:8b"
temperature = 0.0
seed        = 42


# ── chamada ao ollama ─────────────────────────────────────────────────────────

def call_ollama(prompt):
    payload = {
        "model"  : model,
        "prompt" : prompt,
        "stream" : False,
        "options": {
            "temperature": temperature,
            "seed"       : seed,
        },
    }
    resp = requests.post(ollama_url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"]


def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        raise


# ── prompts focados por categoria ─────────────────────────────────────────────
# cada prompt recebe apenas os dados relevantes para aquela categoria
# isto força o modelo a usar os números reais em vez de alucinar

def prompt_trafego(metrics):
    t = metrics["traffic"]
    data = {
        "total_visitantes_semana" : t["total_unique_visitors"],
        "visitantes_por_dia"      : t["visitors_per_day"],
        "visitantes_por_hora"     : t["visitors_per_hour"],
        "hora_pico"               : t["peak_hour"],
        "dia_mais_movimentado"    : t["busiest_day"],
        "dia_menos_movimentado"   : t["quietest_day"],
        "duracao_media_visita_min": t["avg_visit_duration_minutes"],
        "duracao_mediana_min"     : t["median_visit_duration_minutes"],
    }
    return f"""És um especialista em retalho. Analisa estes dados de tráfego de uma loja e gera 2 insights em português.

DADOS DE TRÁFEGO:
{json.dumps(data, indent=2, ensure_ascii=False)}

REGRAS:
- Usa os números exactos acima — não inventes valores
- Cada insight deve ter observacao com números concretos, implicacao operacional e recomendacao executável
- Responde APENAS com JSON válido, sem markdown

Formato:
{{
  "insights": [
    {{
      "categoria": "trafego",
      "titulo": "...",
      "observacao": "... (com números dos dados acima)",
      "implicacao": "...",
      "recomendacao": "...",
      "urgencia": "imediata|esta_semana|proximo_mes",
      "confianca": 0.0
    }}
  ]
}}"""


def prompt_zonas(metrics):
    z = metrics["zones"]
    data = {
        "top_zonas_trafego": z["top_zones_traffic"],
        "top_zonas_dwell"  : z["top_zones_dwell"],
        "top_sequencias"   : z["top_sequences"],
        "detalhes_zonas"   : {k: v for k, v in z["zone_stats"].items()
                              if v["total_visits"] > 500},
    }
    return f"""És um especialista em retalho. Analisa estes dados de zonas de uma loja e gera 2 insights em português.

DADOS DE ZONAS:
{json.dumps(data, indent=2, ensure_ascii=False)}

REGRAS:
- Usa os números exactos acima — não inventes valores
- Foca em zonas com comportamento interessante (dwell alto, tráfego anómalo, sequências frequentes)
- Cada insight deve ter observacao com números concretos, implicacao operacional e recomendacao executável
- Responde APENAS com JSON válido, sem markdown

Formato:
{{
  "insights": [
    {{
      "categoria": "zona",
      "titulo": "...",
      "observacao": "... (com números dos dados acima)",
      "implicacao": "...",
      "recomendacao": "...",
      "urgencia": "imediata|esta_semana|proximo_mes",
      "confianca": 0.0
    }}
  ]
}}"""


def prompt_funil(metrics):
    data = metrics["funnel"]
    return f"""És um especialista em retalho. Analisa estes dados do funil de clientes de uma loja e gera 2 insights em português.

DADOS DO FUNIL:
{json.dumps(data, indent=2, ensure_ascii=False)}

REGRAS:
- Usa os números exactos acima — não inventes valores
- Foca nas taxas de conversão e no perfil dos não-conversores
- Cada insight deve ter observacao com números concretos, implicacao operacional e recomendacao executável
- Responde APENAS com JSON válido, sem markdown

Formato:
{{
  "insights": [
    {{
      "categoria": "funil",
      "titulo": "...",
      "observacao": "... (com números dos dados acima)",
      "implicacao": "...",
      "recomendacao": "...",
      "urgencia": "imediata|esta_semana|proximo_mes",
      "confianca": 0.0
    }}
  ]
}}"""


def prompt_anomalias(metrics):
    # top 5 anomalias por z-score
    top = sorted(metrics["anomalies"]["anomalies"],
                 key=lambda x: -abs(x["z_score"]))[:5]
    data = {
        "total_anomalias": metrics["anomalies"]["total"],
        "threshold_sigma": metrics["anomalies"]["threshold"],
        "top_anomalias"  : top,
    }
    return f"""És um especialista em retalho. Analisa estas anomalias detectadas numa loja e gera 2 insights em português.

ANOMALIAS DETECTADAS (desvio > 2 sigma em relação aos 6 dias anteriores):
{json.dumps(data, indent=2, ensure_ascii=False)}

REGRAS:
- Usa os números exactos acima — não inventes valores
- Para cada anomalia: descreve o que aconteceu, a magnitude, e uma acção concreta
- Cada insight deve ter observacao com números concretos, implicacao operacional e recomendacao executável
- Responde APENAS com JSON válido, sem markdown

Formato:
{{
  "insights": [
    {{
      "categoria": "anomalia",
      "titulo": "...",
      "observacao": "... (com números dos dados acima)",
      "implicacao": "...",
      "recomendacao": "...",
      "urgencia": "imediata|esta_semana|proximo_mes",
      "confianca": 0.0
    }}
  ]
}}"""


def prompt_demografico(metrics):
    d = metrics["demographics"]
    data = {
        "genero_geral_pct": d["gender_overall_pct"],
        "idade_geral_pct" : d["age_overall_pct"],
        "genero_por_hora" : d["gender_by_hour"],
        "idade_por_hora"  : d["age_by_hour"],
    }
    return f"""És um especialista em retalho. Analisa estes dados demográficos de uma loja e gera 2 insights em português.

DADOS DEMOGRÁFICOS:
{json.dumps(data, indent=2, ensure_ascii=False)}

REGRAS:
- Usa os números exactos acima — não inventes valores
- Foca em padrões por hora ou segmentos com comportamento distinto
- Cada insight deve ter observacao com números concretos, implicacao operacional e recomendacao executável
- Responde APENAS com JSON válido, sem markdown

Formato:
{{
  "insights": [
    {{
      "categoria": "demografico",
      "titulo": "...",
      "observacao": "... (com números dos dados acima)",
      "implicacao": "...",
      "recomendacao": "...",
      "urgencia": "imediata|esta_semana|proximo_mes",
      "confianca": 0.0
    }}
  ]
}}"""


def prompt_resumo(insights):
    titulos = [f"- {i['titulo']}: {i['observacao']}" for i in insights]
    return f"""Com base nestes insights de uma loja de retalho, escreve um resumo executivo em português com exactamente 3 bullets.

INSIGHTS:
{chr(10).join(titulos)}

REGRAS:
- Cada bullet deve ser uma frase concisa com o facto mais importante
- Usa números concretos
- Responde APENAS com JSON válido, sem markdown

Formato:
{{
  "resumo_executivo": ["bullet 1", "bullet 2", "bullet 3"]
}}"""


# ── validação ─────────────────────────────────────────────────────────────────

def validate(insights):
    valid_cat = {"trafego", "zona", "funil", "anomalia", "demografico"}
    valid_urg = {"imediata", "esta_semana", "proximo_mes"}
    required  = {"categoria", "titulo", "observacao", "implicacao", "recomendacao", "urgencia", "confianca"}

    for i, ins in enumerate(insights):
        for k in required:
            if k not in ins:
                ins[k] = "" if k != "confianca" else 0.5
        if ins["categoria"] not in valid_cat:
            ins["categoria"] = "trafego"
        if ins["urgencia"] not in valid_urg:
            ins["urgencia"] = "esta_semana"
        try:
            ins["confianca"] = max(0.0, min(1.0, float(ins["confianca"])))
        except (ValueError, TypeError):
            ins["confianca"] = 0.5
        ins["id"] = f"INS_{i+1:03d}"

    return insights


def evaluate(insights):
    def num_density(text):
        tokens = str(text).split()
        return sum(1 for t in tokens if re.search(r"\d", t)) / max(len(tokens), 1)

    def wc(text):
        return len(str(text).split())

    n = max(len(insights), 1)
    return {
        "n_insights"            : len(insights),
        "avg_observacao_words"  : round(sum(wc(i["observacao"])   for i in insights) / n, 1),
        "avg_recomendacao_words": round(sum(wc(i["recomendacao"]) for i in insights) / n, 1),
        "avg_numeric_density"   : round(sum(num_density(i["observacao"]) for i in insights) / n, 3),
        "categoria_coverage"    : len({i["categoria"] for i in insights}),
        "avg_confidence"        : round(sum(i["confianca"] for i in insights) / n, 3),
    }


# ── estratégias ───────────────────────────────────────────────────────────────

def run_focused(metrics):
    """estratégia focused: um prompt por categoria com apenas os dados relevantes"""
    categorias = [
        ("trafego",    prompt_trafego),
        ("zonas",      prompt_zonas),
        ("funil",      prompt_funil),
        ("anomalias",  prompt_anomalias),
        ("demografico",prompt_demografico),
    ]
    all_insights = []
    for nome, fn in categorias:
        print(f"  categoria: {nome} ...", flush=True)
        prompt = fn(metrics)
        try:
            raw  = call_ollama(prompt)
            data = extract_json(raw)
            all_insights.extend(data.get("insights", []))
        except Exception as e:
            print(f"  erro em {nome}: {e}")

    return all_insights


def run_zero_shot(metrics):
    """estratégia zero-shot: um único prompt com todo o json"""
    subset = json.dumps({
        "data_period"   : metrics["data_period"],
        "traffic"       : metrics["traffic"],
        "top_zones"     : metrics["zones"]["top_zones_traffic"],
        "top_sequences" : metrics["zones"]["top_sequences"],
        "funnel"        : metrics["funnel"],
        "demographics"  : {"gender_overall_pct": metrics["demographics"]["gender_overall_pct"],
                           "age_overall_pct"   : metrics["demographics"]["age_overall_pct"]},
        "anomalies"     : metrics["anomalies"],
    }, indent=2, ensure_ascii=False)

    prompt = f"""És um especialista em retalho. Analisa os dados abaixo e gera 10 insights em português.

DADOS:
{subset}

INSTRUÇÕES:
- Usa os números exactos dos dados — não inventes valores
- Cobre as categorias: trafego, zona, funil, anomalia, demografico
- Cada insight com observacao numérica, implicacao e recomendacao executável
- Responde APENAS com JSON válido, sem markdown

{{
  "insights": [
    {{
      "categoria": "trafego|zona|funil|anomalia|demografico",
      "titulo": "...",
      "observacao": "...",
      "implicacao": "...",
      "recomendacao": "...",
      "urgencia": "imediata|esta_semana|proximo_mes",
      "confianca": 0.0
    }}
  ]
}}"""

    raw  = call_ollama(prompt)
    data = extract_json(raw)
    return data.get("insights", [])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--strategy", default="both",
                        choices=["zero_shot", "focused", "both"])
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        metrics = json.load(f)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model"       : model,
        "temperature" : temperature,
        "strategies"  : {},
    }

    to_run = []
    if args.strategy in ("zero_shot", "both"):
        to_run.append("zero_shot")
    if args.strategy in ("focused", "both"):
        to_run.append("focused")

    for name in to_run:
        print(f"\na correr estratégia: {name} ...", flush=True)
        if name == "zero_shot":
            insights = run_zero_shot(metrics)
        else:
            insights = run_focused(metrics)

        insights = validate(insights)
        evals    = evaluate(insights)
        print(f"  resultado: {evals}")

        # resumo executivo
        print(f"  a gerar resumo executivo ...", flush=True)
        try:
            raw_resumo = call_ollama(prompt_resumo(insights))
            resumo     = extract_json(raw_resumo).get("resumo_executivo", [])
        except Exception:
            resumo = [i["titulo"] for i in insights[:3]]

        output["strategies"][name] = {
            "insights"        : insights,
            "resumo_executivo": resumo,
            "eval"            : evals,
        }

    # melhor estratégia por densidade numérica
    if len(output["strategies"]) > 1:
        best_name = max(output["strategies"].items(),
                        key=lambda x: x[1]["eval"]["avg_numeric_density"])[0]
    else:
        best_name = to_run[0]

    output["best_strategy"]    = best_name
    output["insights"]         = output["strategies"][best_name]["insights"]
    output["resumo_executivo"] = output["strategies"][best_name]["resumo_executivo"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nguardado: {args.output}")
    print(f"melhor estratégia: {best_name}")


if __name__ == "__main__":
    main()