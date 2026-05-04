import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import re
import time
from pathlib import Path
import requests
 
# Configuração 
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.1:8b"
TEMPERATURE = 0.0   # reprodutibilidade garantida
 
OUTPUT_SCHEMA = """{
  "insights": [
    {
      "id": "INS_001",
      "categoria": "trafego|zona|funil|anomalia|demografico",
      "titulo": "frase curta que resume o insight",
      "observacao": "o que os dados mostram: factos e números concretos",
      "implicacao": "o que isto significa operacionalmente",
      "recomendacao": "ação concreta que o gestor pode tomar",
      "urgencia": "imediata|esta_semana|proximo_mes",
      "confianca": 0.0
    }
  ],
  "resumo_executivo": "3 bullets com os insights mais importantes"
}"""
 
# Chamada ao Ollama
def call_ollama(prompt: str, system: str = "") -> str:
    """Chama o modelo local via Ollama e devolve o texto gerado."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
 
    payload = {
        "model":  MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 4096,
        }
    }
 
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            "[ERRO] Não foi possível ligar ao Ollama. "
            "Certifica-te que está a correr: ollama serve"
        )
 
 
def extract_json(text: str) -> dict:
    """Extrai o primeiro bloco JSON válido do texto do LLM."""
    # Tentar extrair bloco ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
 
    # Tentar encontrar JSON diretamente no texto
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
 
    # Devolver estrutura vazia com o texto bruto para debug
    return {"insights": [], "resumo_executivo": text[:500], "_raw": text}
 
# Preparação dos dados para o prompt
def prepare_metrics_summary(metrics: dict) -> str:
    """
    Converte metrics.json numa representação textual compacta para o prompt.
    Não envia tudo — seleciona os dados mais relevantes para evitar context overflow.
    """
    t  = metrics["traffic"]
    f  = metrics["funnel"]
    z  = metrics["zones"]
    d  = metrics["demographics"]
    an = metrics["anomalies"]
 
    # Tráfego
    days_str = "\n".join(
        f"  - {d['day_name'].capitalize()} ({d['date']}): {d['visitors']} visitantes"
        for d in t["visitors_by_day"]
    )
    hours_str = ", ".join(
        f"{h['hour']}h={h['visitors']}"
        for h in t["visitors_by_hour"]
    )
 
    # Top zonas por tráfego
    zones_sorted = sorted(z["by_zone"], key=lambda x: x["unique_visitors"], reverse=True)
    top_zones = zones_sorted[:8]
    zones_str = "\n".join(
        f"  - {z['zone_id']} ({z.get('description','')[:30]}): "
        f"{z['unique_visitors']} visitantes, "
        f"dwell médio={z.get('dwell_mean_s','N/A')}s, "
        f"taxa paragem={z.get('stop_rate','N/A')}"
        for z in top_zones
    )
 
    # Sequências mais frequentes
    seqs_str = "\n".join(
        f"  - {s['sequence']}: {s['count']}x"
        for s in z["top_sequences"][:5]
    )
 
    # Funil
    fn = f
    funnel_str = (
        f"  Total visitantes: {fn['total_visitors']}\n"
        f"  Chegaram a corredores: {fn['reached_navigation']['count']} ({fn['reached_navigation']['pct']}%)\n"
        f"  Chegaram a secções: {fn['reached_sections']['count']} ({fn['reached_sections']['pct']}%)\n"
        f"  Chegaram à caixa: {fn['reached_checkout']['count']} ({fn['reached_checkout']['pct']}%)\n"
        f"  Completaram compra (saíram por Z_CK): {fn['completed_purchase']['count']} ({fn['completed_purchase']['pct']}%)"
    )
 
    # Demografia
    gender = d["gender_distribution"]
    age    = d["age_distribution"]
    demo_str = (
        f"  Género: {gender}\n"
        f"  Faixa etária: {age}\n"
        f"  Dwell médio por idade: {d['avg_dwell_by_age']}"
    )
 
    # Anomalias top 5
    anom_str = "\n".join(
        f"  - {a['zone_id']} às {a['hour']}h: {a['visitors_d7']} visitantes "
        f"(média={a['baseline_mean']}, z={a['z_score']}, {a['direction']} da média)"
        for a in an["anomalies"][:5]
    )
 
    return f""" MÉTRICAS DA SEMANA ({metrics['meta']['week_start']} a {metrics['meta']['week_end']})
 
TRÁFEGO:
  Total visitantes semana: {t['total_visitors_week']}
  Dia mais movimentado: {t['busiest_day']['day_name']} com {t['busiest_day']['visitors']} visitantes
  Dia mais calmo: {t['quietest_day']['day_name']} com {t['quietest_day']['visitors']} visitantes
  Hora de pico: {t['peak_hour']}h
  Duração média visita: {t['avg_visit_duration_min']} minutos
  Visitantes por dia:
{days_str}
  Visitantes por hora: {hours_str}
 
ZONAS (top 8 por tráfego):
{zones_str}
 
SEQUÊNCIAS MAIS FREQUENTES:
{seqs_str}
 
FUNIL DE CLIENTE:
{funnel_str}
 
DEMOGRAFIA:
{demo_str}
 
ANOMALIAS DIA 7 (z > 2):
  Total anomalias: {an['total_anomalies']}
{anom_str}
"""
 
# Estratégia A — Zero-shot 
SYSTEM_PROMPT = """És um consultor especialista em retalho e análise de comportamento de clientes em loja.
A tua função é analisar métricas de uma loja de retalho e gerar insights acionáveis em português europeu para o gestor de loja.
Respondes SEMPRE e APENAS com JSON válido, sem texto adicional antes ou depois."""
 
 
def prompt_zero_shot(metrics_text: str) -> str:
    return f"""Analisa as seguintes métricas de uma loja de retalho e gera exatamente 6 insights acionáveis.
 
{metrics_text}
 
Responde APENAS com JSON válido seguindo este schema exato:
{OUTPUT_SCHEMA}
 
Regras:
- Cada insight deve ter números concretos retirados dos dados
- As recomendações devem ser específicas e executáveis
- A confiança deve ser entre 0.0 e 1.0
- Responde em português europeu
- Não incluas texto fora do JSON"""
 
 
# Estratégia B — Few-shot
FEW_SHOT_EXAMPLES = """
 EXEMPLOS DE INSIGHTS 
 
MAU INSIGHT (genérico, vago — NÃO fazer assim):
{
  "titulo": "A zona de frescos teve bastante tráfego",
  "observacao": "Muitas pessoas visitaram esta zona",
  "implicacao": "É uma zona popular",
  "recomendacao": "Melhorar o atendimento ao cliente"
}
 
BOM INSIGHT (específico, com números, acionável — fazer assim):
{
  "titulo": "Z_S1 (frescos) tem a maior taxa de paragem da loja",
  "observacao": "Z_S1 registou 847 visitantes na semana com taxa de paragem de 0.82, a mais alta de todas as secções de produto",
  "implicacao": "Os clientes passam tempo significativo nesta secção — é um ponto de decisão de compra crítico",
  "recomendacao": "Colocar promoções e produtos de margem alta em destaque em Z_S1; considerar aumentar stock de frescos às sextas e sábados quando o tráfego é 40% acima da média"
}
 
MAU INSIGHT (anomalia vaga — NÃO fazer assim):
{
  "titulo": "Houve uma anomalia na loja",
  "observacao": "O tráfego foi diferente do normal",
  "recomendacao": "Investigar o que aconteceu"
}
 
BOM INSIGHT (anomalia específica — fazer assim):
{
  "titulo": "Queda anómala em Z_N4 no domingo às 16h",
  "observacao": "Z_N4 teve 0 visitantes às 16h do dia 7 (domingo), contra uma média de 23 nos mesmos dias/horas anteriores (z=-3.2)",
  "implicacao": "Possível obstrução física, problema de sinalização ou encerramento temporário do corredor",
  "recomendacao": "Verificar registos de incidentes do domingo às 16h; inspecionar sinalização de Z_N4; se recorrente, instalar câmara de supervisão neste corredor"
}
 FIM DOS EXEMPLOS
"""
 
 
def prompt_few_shot(metrics_text: str) -> str:
    return f"""Analisa as seguintes métricas de uma loja de retalho e gera exatamente 6 insights acionáveis.
 
{FEW_SHOT_EXAMPLES}
 
Segue o estilo dos BOM INSIGHT acima: sempre com números concretos, causas plausíveis e recomendações executáveis.
Evita o estilo dos MAU INSIGHT: genérico, vago, sem números.
 
DADOS DA SEMANA:
{metrics_text}
 
Responde APENAS com JSON válido seguindo este schema exato:
{OUTPUT_SCHEMA}
 
Regras:
- Cada insight deve citar números específicos dos dados acima
- As recomendações devem ser concretas e executáveis sem interpretação adicional
- A confiança deve ser entre 0.0 e 1.0
- Responde em português europeu
- Não incluas texto fora do JSON"""
 
# Comparação A vs B
def score_insight(insight: dict) -> dict:
    """
    Avalia quantitativamente um insight em 3 dimensões:
      - especificidade: tem números concretos?
      - acionabilidade: a recomendação é específica?
      - completude: tem todos os campos preenchidos?
    """
    obs  = insight.get("observacao", "")
    rec  = insight.get("recomendacao", "")
    impl = insight.get("implicacao", "")
 
    # Contar números no texto
    numbers = len(re.findall(r'\d+[\.,]?\d*', obs + rec))
    specificity = min(1.0, numbers / 5)  # 5+ números = score máximo
 
    # Acionabilidade: verbos de ação concretos
    action_verbs = ["abrir", "fechar", "colocar", "aumentar", "reduzir",
                    "mover", "verificar", "instalar", "reforçar", "ajustar",
                    "contratar", "promover", "reorganizar"]
    has_action = any(v in rec.lower() for v in action_verbs)
    actionability = 1.0 if has_action else 0.3
 
    # Completude: todos os campos preenchidos e não vazios
    required = ["titulo", "observacao", "implicacao", "recomendacao", "urgencia"]
    completeness = sum(1 for f in required if insight.get(f, "").strip()) / len(required)
 
    total = (specificity + actionability + completeness) / 3
 
    return {
        "specificity":   round(specificity, 2),
        "actionability": round(actionability, 2),
        "completeness":  round(completeness, 2),
        "total":         round(total, 2),
    }
 
 
def compare_strategies(result_a: dict, result_b: dict) -> dict:
    """Compara quantitativamente os resultados das duas estratégias."""
    scores_a = [score_insight(i) for i in result_a.get("insights", [])]
    scores_b = [score_insight(i) for i in result_b.get("insights", [])]
 
    def avg(scores, key):
        vals = [s[key] for s in scores]
        return round(sum(vals) / len(vals), 2) if vals else 0
 
    return {
        "strategy_A_zero_shot": {
            "n_insights":    len(scores_a),
            "avg_specificity":   avg(scores_a, "specificity"),
            "avg_actionability": avg(scores_a, "actionability"),
            "avg_completeness":  avg(scores_a, "completeness"),
            "avg_total":         avg(scores_a, "total"),
        },
        "strategy_B_few_shot": {
            "n_insights":    len(scores_b),
            "avg_specificity":   avg(scores_b, "specificity"),
            "avg_actionability": avg(scores_b, "actionability"),
            "avg_completeness":  avg(scores_b, "completeness"),
            "avg_total":         avg(scores_b, "total"),
        },
    }
 
# Main
def main():
    parser = argparse.ArgumentParser(description="Insights — LLM Insight Engine")
    parser.add_argument("--input",    required=True, help="Caminho para metrics.json")
    parser.add_argument("--output",   required=True, help="Caminho para insights.json")
    parser.add_argument("--strategy", default="both",
                        choices=["A", "B", "both"],
                        help="Estratégia de prompting: A (zero-shot), B (few-shot), both")
    args = parser.parse_args()
 
    print(f"[1/4] A carregar {args.input}")
    with open(args.input, encoding="utf-8") as f:
        metrics = json.load(f)
 
    print("[2/4] A preparar sumário de métricas para o prompt")
    metrics_text = prepare_metrics_summary(metrics)
 
    result_a, result_b = None, None
 
    if args.strategy in ("A", "both"):
        print(f"\n[3/4] A executar Estratégia A (zero-shot) com {MODEL}")
        t0 = time.time()
        raw_a = call_ollama(prompt_zero_shot(metrics_text), SYSTEM_PROMPT)
        elapsed_a = round(time.time() - t0, 1)
        result_a  = extract_json(raw_a)
        n_a = len(result_a.get("insights", []))
        print(f"      {n_a} insights gerados em {elapsed_a}s")
 
    if args.strategy in ("B", "both"):
        print(f"\n[3/4] A executar Estratégia B (few-shot) com {MODEL}")
        t0 = time.time()
        raw_b = call_ollama(prompt_few_shot(metrics_text), SYSTEM_PROMPT)
        elapsed_b = round(time.time() - t0, 1)
        result_b  = extract_json(raw_b)
        n_b = len(result_b.get("insights", []))
        print(f"      {n_b} insights gerados em {elapsed_b}s")
 
    # Selecionar o melhor resultado para o output principal
    # (few-shot quando disponível, caso contrário zero-shot)
    primary = result_b if result_b else result_a
 
    # Comparação quantitativa
    comparison = {}
    if result_a and result_b:
        print("\n[4/4] A comparar estratégias")
        comparison = compare_strategies(result_a, result_b)
        print("\nComparação A vs B")
        for strategy, scores in comparison.items():
            print(f"\n  {strategy}:")
            for k, v in scores.items():
                print(f"    {k:<25} {v}")
 
    # Output final
    output = {
        "meta": {
            "model":    MODEL,
            "strategy": args.strategy,
            "input":    args.input,
        },
        "primary_insights": primary,
        "strategy_A_zero_shot": result_a,
        "strategy_B_few_shot":  result_b,
        "comparison": comparison,
    }
 
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
 
    print(f"\n  insights.json escrito em {args.output}")
 
    # Mostrar resumo executivo
    if primary and primary.get("resumo_executivo"):
        print("\nResumo executivo")
        print(primary["resumo_executivo"])
 
 
if __name__ == "__main__":
    main()