import argparse
import json
import re
import time
from datetime import datetime

import requests

# ─── Parâmetros configuráveis ─────────────────────────────────────────────────

config = {
    "ollama_url":  "http://localhost:11434/api/generate",
    "model":       "llama3.1:8b",
    "timeout_s":   180,
    "temperature": 0.0,
    "seed":        42,
}

# ─── Prompts ──────────────────────────────────────────────────────────────────

system_instruction = """\
És um analista de retalho especialista em comportamento de cliente em loja física.
Recebes métricas pré-calculadas de uma semana de operação de um supermercado português.
Responde SEMPRE em português europeu.
Responde APENAS com JSON válido. Sem texto antes ou depois. Sem blocos de código markdown.\
"""

output_schema = """\
O teu output deve ser EXACTAMENTE este JSON e nada mais.

REGRAS OBRIGATÓRIAS:
- O campo "categoria" deve ser UMA destas palavras exactamente: trafego, zona, funil, anomalia, demografico
- O campo "urgencia" deve ser UMA destas palavras exactamente: imediata, esta_semana, proximo_mes
- Gera exactamente 10 insights sem repetições
- Distribui pelas 5 categorias: 2 trafego, 2 zona, 1 funil, 2 anomalia, 2 demografico, 1 livre
- O resumo_executivo deve ter exactamente 3 bullets com números concretos
- Cada observacao deve conter pelo menos um número concreto dos dados

{
  "insights": [
    {
      "id": "INS_001",
      "categoria": "trafego",
      "titulo": "frase curta que resume o insight",
      "observacao": "o que os dados mostram com factos e números concretos",
      "implicacao": "o que isto significa operacionalmente para a loja",
      "recomendacao": "acção concreta e específica que o gestor pode tomar",
      "urgencia": "esta_semana",
      "confianca": 0.9
    }
  ],
  "resumo_executivo": [
    "bullet 1 com número concreto",
    "bullet 2 com número concreto",
    "bullet 3 com número concreto"
  ]
}\
"""

few_shot_examples = """\
EXEMPLOS DE QUALIDADE — segue este nível de detalhe:

--- BOM insight (categoria: trafego) ---
{
  "categoria": "trafego",
  "titulo": "Sábado concentra 17.8% do tráfego semanal",
  "observacao": "Sábado teve 3229 visitantes, 56% acima da terça-feira (2062), o dia mais calmo.",
  "implicacao": "A loja opera em dois regimes distintos: dias úteis com ~2400 visitantes e fim de semana com ~3000.",
  "recomendacao": "Reforçar equipa ao sábado de manhã e rever plano de reposição para sexta à tarde.",
  "urgencia": "esta_semana",
  "confianca": 0.95
}

--- BOM insight (categoria: zona) ---
{
  "categoria": "zona",
  "titulo": "Z_C2 é a zona mais visitada com 9540 visitantes",
  "observacao": "Z_C2 (caixas centrais) recebeu 9540 visitantes com dwell médio de 24.6s. Z_S2 (padaria) foi a menos visitada com apenas 661 visitantes.",
  "implicacao": "As caixas centrais são o principal ponto de passagem. A padaria tem tráfego muito baixo para a sua localização.",
  "recomendacao": "Investigar se a sinalização para Z_S2 é adequada. Considerar promoção de padaria junto às caixas.",
  "urgencia": "proximo_mes",
  "confianca": 0.85
}

--- BOM insight (categoria: funil) ---
{
  "categoria": "funil",
  "titulo": "31% dos visitantes saem sem passar pela caixa",
  "observacao": "De 18142 visitantes totais, 12528 chegaram à caixa (69.1% de conversão). Os 5614 restantes saíram sem comprar.",
  "implicacao": "Quase 1 em cada 3 visitantes não converte. Perda potencial significativa de receita.",
  "recomendacao": "Analisar os percursos dos não-compradores. Testar promoções de impulso junto às saídas.",
  "urgencia": "proximo_mes",
  "confianca": 0.8
}

--- BOM insight (categoria: anomalia) ---
{
  "categoria": "anomalia",
  "titulo": "Z_N1 registou queda de 9σ às 13h no domingo",
  "observacao": "Z_N1 teve 18 visitantes às 13h do domingo contra uma média de 30.3 nos dias anteriores — desvio de -9.03σ.",
  "implicacao": "Queda tão abrupta sugere obstrução física (reposição de stock, paletes) ou falha de câmara.",
  "recomendacao": "Verificar registos de operações em Z_N1 às 13h do domingo. Rever plano de reposição para não bloquear corredores em hora de pico.",
  "urgencia": "imediata",
  "confianca": 0.9
}

--- BOM insight (categoria: demografico) ---
{
  "categoria": "demografico",
  "titulo": "Adultos femininos são o segmento dominante",
  "observacao": "O segmento feminino adulto representa 2738 visitantes, o maior grupo. O segmento masculino adulto tem 2659 visitantes.",
  "implicacao": "A loja tem uma base de clientes maioritariamente adulta. Campanhas devem priorizar este segmento.",
  "recomendacao": "Adaptar promoções e comunicação visual ao perfil adulto feminino, especialmente nas secções Z_S1 e Z_S2.",
  "urgencia": "proximo_mes",
  "confianca": 0.8
}

--- MAU insight (genérico, sem números) ---
{
  "categoria": "trafego",
  "titulo": "A loja teve bastante tráfego esta semana",
  "observacao": "A loja esteve movimentada durante a semana",
  "recomendacao": "Melhorar o atendimento"
}
NUNCA geres insights como este último exemplo.\
"""

# ─── Chamada ao LLM ───────────────────────────────────────────────────────────

def call_ollama(prompt: str) -> str:
    payload = {
        "model":  config["model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config["temperature"],
            "seed":        config["seed"],
        },
    }
    try:
        r = requests.post(config["ollama_url"], json=payload, timeout=config["timeout_s"])
        r.raise_for_status()
        return r.json()["response"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Não foi possível ligar ao Ollama.\n"
            "Certifica-te que está a correr: ollama serve"
        )

def parse_json_response(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "LLM não devolveu JSON válido", "raw": raw[:300]}

# ─── Preparação dos dados para o LLM ─────────────────────────────────────────

def summarise_metrics(metrics: dict) -> str:
    traffic   = metrics["traffic"]
    funnel    = metrics["funnel"]
    anomalies = metrics["anomalies"]

    summary = {
        "trafego": {
            "total_visitantes":          traffic["total_unique_visitors"],
            "dia_mais_movimentado":      traffic["busiest_day"],
            "dia_mais_calmo":            traffic["quietest_day"],
            "duracao_media_visita_min":  traffic["avg_visit_duration_min"],
            "visitantes_por_dia":        traffic["visitors_by_day"],
        },
        "zonas": {
            "top_5_trafego":     metrics["zones"]["top_5_traffic"],
            "bottom_5_trafego":  metrics["zones"]["bottom_5_traffic"],
            "top_10_sequencias": metrics["zones"]["top_10_sequences"],
        },
        "funil": funnel,
        "anomalias": {
            "data_teste":     anomalies["test_date"],
            "total":          anomalies["total_anomalies"],
            "top_10":         anomalies["anomalies"][:10],
        },
        "demograficos": {
            "segmentos": metrics["demographics"]["visitor_segments"][:10],
        },
    }
    return json.dumps(summary, ensure_ascii=False, indent=2, default=str)

# ─── Avaliação da qualidade ───────────────────────────────────────────────────

def score_insights(insights: list) -> dict:
    if not insights:
        return {"n_insights": 0, "specificity": 0.0, "completeness": 0.0,
                "category_coverage": 0.0, "total_score": 0.0}

    required_fields = ["titulo", "observacao", "implicacao", "recomendacao", "urgencia", "confianca"]
    valid_categories = {"trafego", "zona", "funil", "anomalia", "demografico"}

    has_numbers = sum(
        1 for i in insights
        if re.search(r'\d+[.,]?\d*', i.get("observacao", ""))
    )
    specificity = round(has_numbers / len(insights), 2)

    filled = sum(
        sum(1 for f in required_fields if i.get(f)) / len(required_fields)
        for i in insights
    )
    completeness = round(filled / len(insights), 2)

    found_cats = {i.get("categoria", "") for i in insights}
    category_coverage = round(len(found_cats & valid_categories) / len(valid_categories), 2)

    total = round((specificity + completeness + category_coverage) / 3, 2)

    return {
        "n_insights":        len(insights),
        "specificity":       specificity,
        "completeness":      completeness,
        "category_coverage": category_coverage,
        "total_score":       total,
    }

# ─── Dois prompts ─────────────────────────────────────────────────────────────

def build_zero_shot_prompt(metrics_summary: str) -> str:
    return f"""{system_instruction}

{output_schema}

DADOS DA SEMANA:
{metrics_summary}"""

def build_few_shot_prompt(metrics_summary: str) -> str:
    return f"""{system_instruction}

{few_shot_examples}

Agora gera 10 insights de ALTA QUALIDADE para os dados desta semana, seguindo exactamente o nível de detalhe dos bons exemplos acima.

{output_schema}

DADOS DA SEMANA:
{metrics_summary}"""

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="output/metrics.json")
    parser.add_argument("--output", default="output/insights.json")
    parser.add_argument("--model",  default=config["model"])
    args = parser.parse_args()

    config["model"] = args.model

    print(f"[1/3] A carregar métricas de '{args.input}'")
    with open(args.input, encoding="utf-8") as f:
        metrics = json.load(f)

    metrics_summary = summarise_metrics(metrics)
    print(f"      Resumo: {len(metrics_summary)} caracteres para o LLM.")

    results = {}

    for name, build_prompt in [("zero_shot", build_zero_shot_prompt),
                                ("few_shot",  build_few_shot_prompt)]:
        print(f"\n[2/3] A invocar LLM — estratégia '{name}'")
        prompt  = build_prompt(metrics_summary)
        t0      = time.time()
        raw     = call_ollama(prompt)
        elapsed = round(time.time() - t0, 1)
        print(f"      Resposta em {elapsed}s ({len(raw)} caracteres)")

        parsed   = parse_json_response(raw)
        insights = parsed.get("insights", [])
        scores   = score_insights(insights)

        print(f"      Insights gerados : {scores['n_insights']}")
        print(f"      Score total      : {scores['total_score']}  "
              f"(especif. {scores['specificity']}, "
              f"complet. {scores['completeness']}, "
              f"cobertura {scores['category_coverage']})")

        results[name] = {
            "prompt_chars":    len(prompt),
            "response_time_s": elapsed,
            "parsed_output":   parsed,
            "quality_scores":  scores,
        }

    zs = results["zero_shot"]["quality_scores"]["total_score"]
    fs = results["few_shot"]["quality_scores"]["total_score"]
    winner = "few_shot" if fs >= zs else "zero_shot"

    comparison = {
        "winner":          winner,
        "zero_shot_score": zs,
        "few_shot_score":  fs,
        "delta":           round(fs - zs, 2),
        "improvement_pct": round(100 * (fs - zs) / max(zs, 0.01), 1),
    }

    print(f"\n[3/3] Comparação:")
    print(f"      Zero-shot : {zs}")
    print(f"      Few-shot  : {fs}")
    print(f"      Vencedor  : {winner}  (Δ {comparison['delta']})")

    output = {
        "generated_at":  datetime.now().isoformat(),
        "model":         config["model"],
        "strategies":    results,
        "comparison":    comparison,
        "best_insights": results[winner]["parsed_output"],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\ninsights.json escrito em '{args.output}'")

if __name__ == "__main__":
    main()