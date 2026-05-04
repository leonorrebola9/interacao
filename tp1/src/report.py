import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import time
from pathlib import Path
import requests
 
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "llama3.1:8b"
TEMPERATURE = 0.0
 
 
def call_ollama(prompt: str, system: str = "") -> str:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    payload = {
        "model":  MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": 8192}
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise SystemExit("[ERRO] Não foi possível ligar ao Ollama. Corre: ollama serve")
 
 
def prepare_insights_context(insights_data: dict) -> str:
    primary  = insights_data.get("primary_insights", {})
    insights = primary.get("insights", [])
    resumo   = primary.get("resumo_executivo", [])
 
    insights_str = ""
    for ins in insights:
        insights_str += f"""
ID: {ins.get('id')}
Título: {ins.get('titulo')}
Observação: {ins.get('observacao')}
Implicação: {ins.get('implicacao')}
Recomendação: {ins.get('recomendacao')}
Urgência: {ins.get('urgencia')}
---"""
 
    resumo_str = "\n".join(f"• {r}" for r in resumo) if isinstance(resumo, list) else str(resumo)
 
    return f"""RESUMO EXECUTIVO:
{resumo_str}
 
INSIGHTS DETALHADOS:
{insights_str}"""
 
 
def prepare_metrics_context(metrics: dict) -> str:
    t  = metrics.get("traffic", {})
    f  = metrics.get("funnel", {})
    z  = metrics.get("zones", {})
    an = metrics.get("anomalies", {})
 
    days_str = "\n".join(
        f"  - {d['day_name'].capitalize()}: {d['visitors']} visitantes"
        for d in t.get("visitors_by_day", [])
    )
    zones_sorted = sorted(z.get("by_zone", []), key=lambda x: x.get("unique_visitors", 0), reverse=True)
    zones_str = "\n".join(
        f"  - {z['zone_id']}: {z['unique_visitors']} visitantes, dwell médio={z.get('dwell_mean_s','N/A')}s"
        for z in zones_sorted[:8]
    )
    anom_str = "\n".join(
        f"  - {a['zone_id']} às {a['hour']}h: {a['visitors_d7']} visitantes (média esperada={a['baseline_mean']})"
        for a in an.get("anomalies", [])[:3]
    )
 
    fn = f
    return f""" NÚMEROS REAIS DA SEMANA — usa APENAS estes valores, nunca inventes outros 
 
TRÁFEGO:
  Total de visitantes na semana: {t.get('total_visitors_week')}
  Dia mais movimentado: {t.get('busiest_day', {}).get('day_name')} com {t.get('busiest_day', {}).get('visitors')} visitantes
  Dia mais calmo: {t.get('quietest_day', {}).get('day_name')} com {t.get('quietest_day', {}).get('visitors')} visitantes
  Hora de pico: {t.get('peak_hour')}h
  Duração média de visita: {t.get('avg_visit_duration_min')} minutos
  Visitantes por dia:
{days_str}
 
TOP ZONAS (por número de visitantes):
{zones_str}
 
FUNIL:
  Total visitantes: {fn.get('total_visitors')}
  Chegaram a corredores: {fn.get('reached_navigation', {}).get('count')} ({fn.get('reached_navigation', {}).get('pct')}%)
  Chegaram a secções de produto: {fn.get('reached_sections', {}).get('count')} ({fn.get('reached_sections', {}).get('pct')}%)
  Chegaram à caixa: {fn.get('reached_checkout', {}).get('count')} ({fn.get('reached_checkout', {}).get('pct')}%)
  Completaram compra: {fn.get('completed_purchase', {}).get('count')} ({fn.get('completed_purchase', {}).get('pct')}%)
 
ANOMALIAS (top 3):
{anom_str}
 FIM DOS NÚMEROS REAIS """
 
 
SYSTEM_REPORT = """És um consultor especialista em retalho. Escreves briefings semanais
para gestores de loja sem formação técnica, em português europeu, com linguagem direta e
números concretos. Nunca uses termos como 'z-score', 'dataset' ou 'pipeline'.
REGRA FUNDAMENTAL: Usa APENAS os números fornecidos. Nunca inventes ou estimes valores."""
 
 
def build_report_prompt(insights_ctx: str, metrics_ctx: str, week_start: str, week_end: str) -> str:
    return f"""Escreve o briefing semanal completo da loja para a semana de {week_start} a {week_end}.
 
{metrics_ctx}
 
{insights_ctx}
 
Escreve em Markdown com EXATAMENTE estas 6 secções:
 
# Briefing Semanal — {week_start} a {week_end}
 
## 1. Resumo Executivo
(máximo 150 palavras — os 3 factos mais importantes, com números reais)
 
## 2. Performance de Tráfego
(total de visitantes, visitantes por dia, hora de pico — usa apenas os números fornecidos)
 
## 3. Análise de Zonas
(top 3 zonas; zonas problemáticas com hipótese de causa e recomendação)
 
## 4. Funil de Clientes
(percentagens reais do funil; onde se perde tráfego; perfil de quem não chega à caixa)
 
## 5. Anomalias da Semana
(descreve as anomalias; possível causa; ação recomendada — sem usar 'z-score')
 
## 6. Recomendações para a Próxima Semana
(máximo 5 ações concretas, ordenadas por urgência)
 
REGRAS: Usa APENAS os números fornecidos. Português europeu. Sem termos técnicos."""
 
 
def add_report_header(report_md: str, meta: dict) -> str:
    return f"""---
gerado_em: {time.strftime('%Y-%m-%d %H:%M')}
modelo: {meta.get('model', MODEL)}
fonte: {meta.get('input', 'insights.json')}
---
 
""" + report_md
 
 
def clean_report(text: str) -> str:
    text = text.replace("```markdown", "").replace("```", "")
    if not text.strip().startswith("#"):
        idx = text.find("#")
        if idx > 0:
            text = text[idx:]
    return text.strip()
 
 
def main():
    parser = argparse.ArgumentParser(description="Report — geração do briefing semanal")
    parser.add_argument("--input",   required=True, help="Caminho para insights.json")
    parser.add_argument("--metrics", default=None,  help="Caminho para metrics.json (recomendado)")
    parser.add_argument("--output",  required=True, help="Caminho para weekly_report.md")
    args = parser.parse_args()
 
    print(f"[1/3] A carregar {args.input}")
    with open(args.input, encoding="utf-8") as f:
        insights_data = json.load(f)
    meta = insights_data.get("meta", {})
 
    metrics_context = ""
    if args.metrics:
        print(f"      A carregar métricas de {args.metrics}")
        with open(args.metrics, encoding="utf-8") as f:
            metrics = json.load(f)
        metrics_context = prepare_metrics_context(metrics)
    else:
        print("      [AVISO] --metrics não fornecido. Recomendado para evitar alucinações.")
 
    week_start = "10 de março de 2025"
    week_end   = "16 de março de 2025"
 
    print("[2/3] A gerar relatório com LLM")
    insights_context = prepare_insights_context(insights_data)
    prompt = build_report_prompt(insights_context, metrics_context, week_start, week_end)
 
    t0 = time.time()
    raw_report = call_ollama(prompt, SYSTEM_REPORT)
    elapsed = round(time.time() - t0, 1)
    print(f"      Relatório gerado em {elapsed}s")
 
    print(f"[3/3] A escrever {args.output}")
    report_md = clean_report(raw_report)
    report_md = add_report_header(report_md, meta)
 
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_md)
 
    print("\nPreview")
    print("\n".join(report_md.split("\n")[:20]))
    print(f"\n  Relatório completo em {args.output}")
 
 
if __name__ == "__main__":
    main()