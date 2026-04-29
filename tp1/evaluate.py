import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import pandas as pd
 
# Configuração
# Caminhos dos módulos do pipeline (relativos à raiz do projeto)
SRC_DIR     = Path("src")
STITCHER    = SRC_DIR / "stitcher.py"
ANALYTICS   = SRC_DIR / "analytics.py"
INSIGHTS    = SRC_DIR / "insights.py"
ZONES_FILE  = Path("data") / "zones.json"
 
# Execução dos módulos do pipeline
def run_module(script: Path, args: list[str], step: str) -> bool:
    """Corre um módulo do pipeline e devolve True se correu sem erros."""
    cmd = [sys.executable, str(script)] + args
    print(f"\n  [{step}] A correr: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)
 
    if result.returncode != 0:
        print(f"  [ERRO] {step} falhou após {elapsed}s:")
        print(result.stderr[-500:])
        return False
 
    print(f"  [OK] {step} concluído em {elapsed}s")
    return True
 
# Métricas de qualidade do stitching
def evaluate_stitching(events_path: str, journeys_path: str) -> dict:
    """Calcula as 3 métricas de qualidade do stitching."""
    events   = pd.read_csv(events_path, parse_dates=["timestamp"])
    journeys = pd.read_csv(journeys_path, parse_dates=["entry_time", "exit_time"])
 
    total_events = len(events)
    n_trajs      = journeys["person_id"].nunique()
 
    # Consistência: sem sobreposição temporal
    overlaps = 0
    for pid, group in journeys.groupby("person_id"):
        group = group.sort_values("entry_time")
        entries = group["entry_time"].values
        exits   = group["exit_time"].fillna(group["entry_time"]).values
        for i in range(len(entries) - 1):
            if exits[i] is not None and entries[i + 1] < exits[i]:
                overlaps += 1
    consistency = 1.0 - (overlaps / max(1, n_trajs))
 
    # Cobertura: eventos atribuídos / total
    assigned_events = len(journeys) * 3  # aprox.: entry + linger + exit
    coverage = min(1.0, assigned_events / total_events)
 
    # Completude: início em Z_E, fim em Z_E ou Z_CK
    entrance_zones = {"Z_E1", "Z_E2"}
    exit_zones     = {"Z_E1", "Z_E2", "Z_CK"}
    first_zones    = journeys.groupby("person_id")["zone_id"].first()
    last_zones     = journeys.groupby("person_id")["zone_id"].last()
    starts_ok      = first_zones.isin(entrance_zones).sum()
    ends_ok        = last_zones.isin(exit_zones).sum()
    completeness   = (starts_ok + ends_ok) / (2 * n_trajs) if n_trajs > 0 else 0
 
    # Gap distribution
    gaps = []
    for pid, group in journeys.groupby("person_id"):
        group   = group.sort_values("entry_time")
        exits   = group["exit_time"].tolist()
        entries = group["entry_time"].tolist()
        for i in range(len(entries) - 1):
            if exits[i] is not None:
                gap = (entries[i + 1] - exits[i]).total_seconds()
                gaps.append(gap)
 
    gap_s = pd.Series(gaps)
 
    return {
        "trajectories_total": int(n_trajs),
        "events_total":       int(total_events),
        "consistency":        round(consistency, 4),
        "consistency_pct":    f"{consistency * 100:.1f}%",
        "coverage":           round(coverage, 4),
        "coverage_pct":       f"{coverage * 100:.1f}%",
        "completeness":       round(completeness, 4),
        "completeness_pct":   f"{completeness * 100:.1f}%",
        "overlaps_found":     int(overlaps),
        "gap_median_s":       round(gap_s.median(), 1) if len(gaps) > 0 else None,
        "gap_p95_s":          round(gap_s.quantile(0.95), 1) if len(gaps) > 0 else None,
    }
 
# Deteção de anomalias injetadas
def evaluate_anomaly_detection(insights_path: str, metrics_path: str) -> dict:
    """
    Verifica quantas anomalias foram detetadas nos insights.
    Como o professor injeta anomalias conhecidas, este método verifica
    se o sistema as identificou corretamente.
 
    Na ausência de anomalias conhecidas (dataset de treino), reporta
    apenas as métricas gerais de deteção.
    """
    with open(insights_path, encoding="utf-8") as f:
        insights_data = json.load(f)
 
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
 
    # Anomalias calculadas pelo analytics.py
    total_anomalies_detected = metrics.get("anomalies", {}).get("total_anomalies", 0)
    anomalies_list = metrics.get("anomalies", {}).get("anomalies", [])
 
    # Insights com categoria "anomalia"
    primary   = insights_data.get("primary_insights", {})
    insights  = primary.get("insights", [])
    anomaly_insights = [
        i for i in insights
        if "anomalia" in i.get("categoria", "").lower()
    ]
 
    # Verificação de precisão numérica:
    # Os números nos insights devem ser verificáveis no metrics.json
    numeric_checks = 0
    numeric_ok     = 0
    for ins in insights:
        obs = ins.get("observacao", "")
        # Verificar se os valores de visitantes citados existem nos dados
        import re
        numbers_in_obs = re.findall(r'\b(\d+)\b', obs)
        for n in numbers_in_obs:
            n_int = int(n)
            if n_int > 10:  # ignorar números muito pequenos
                numeric_checks += 1
                # Verificar se o número aparece em alguma métrica
                metrics_str = json.dumps(metrics)
                if str(n_int) in metrics_str:
                    numeric_ok += 1
 
    precision = numeric_ok / numeric_checks if numeric_checks > 0 else None
 
    return {
        "anomalies_in_metrics":      int(total_anomalies_detected),
        "anomaly_insights_generated": len(anomaly_insights),
        "numeric_precision":         round(precision, 3) if precision else None,
        "numeric_precision_pct":     f"{precision * 100:.1f}%" if precision else "N/A",
        "numeric_checks_total":      numeric_checks,
        "numeric_checks_ok":         numeric_ok,
        "top_anomalies": [
            {
                "zone_id":   a.get("zone_id"),
                "hour":      a.get("hour"),
                "z_score":   a.get("z_score"),
                "direction": a.get("direction"),
            }
            for a in anomalies_list[:5]
        ],
    }
 
# Verificação de alucinação no report
def evaluate_hallucination(report_path: str, metrics_path: str) -> dict:
    """
    Verifica se os números no relatório semanal são verificáveis no metrics.json.
    Métrica: % de valores numéricos no report que existem nos dados reais.
    """
    if not Path(report_path).exists():
        return {"error": "weekly_report.md não encontrado"}
 
    import re
    with open(report_path, encoding="utf-8") as f:
        report_text = f.read()
 
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
 
    metrics_str = json.dumps(metrics)
 
    # Extrair todos os números do relatório (excluindo anos e datas)
    numbers = re.findall(r'\b(\d{2,5})\b', report_text)
    checks  = 0
    ok      = 0
    for n in numbers:
        n_int = int(n)
        if 10 <= n_int <= 99999:  # range plausível para métricas de loja
            checks += 1
            if str(n_int) in metrics_str:
                ok += 1
 
    hallucination_rate = 1.0 - (ok / checks) if checks > 0 else None
 
    return {
        "numbers_checked":    checks,
        "numbers_verified":   ok,
        "hallucination_rate": round(hallucination_rate, 3) if hallucination_rate is not None else None,
        "hallucination_pct":  f"{hallucination_rate * 100:.1f}%" if hallucination_rate is not None else "N/A",
        "note": "Taxa de alucinação = % de números no relatório não verificáveis nos dados reais",
    }
 
# Main
def main():
    parser = argparse.ArgumentParser(description="Harness de Avaliação do Pipeline")
    parser.add_argument("--data",   required=True, help="Caminho para events_validation.csv")
    parser.add_argument("--output", required=True, help="Caminho para evaluation_report.json")
    parser.add_argument("--zones",  default=str(ZONES_FILE), help="Caminho para zones.json")
    args = parser.parse_args()
 
    print("     Harness de avaliação    ")
    t_total = time.time()
 
    # Criar diretório temporário para os outputs intermédios
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        journeys_path = str(tmp / "journeys.csv")
        metrics_path  = str(tmp / "metrics.json")
        insights_path = str(tmp / "insights.json")
        report_path   = str(tmp / "weekly_report.md")
 
        # Passo 1: Stitching
        print("\n[1/4] Stitching de trajetórias")
        ok = run_module(STITCHER, [
            "--input",  args.data,
            "--output", journeys_path,
            "--zones",  args.zones,
        ], "stitcher.py")
        if not ok:
            print("[FATAL] stitcher.py falhou. A avaliar o que foi produzido...")
 
        # Passo 2: Analytics
        print("\n[2/4] Cálculo de métricas analíticas")
        ok = run_module(ANALYTICS, [
            "--input",  journeys_path,
            "--output", metrics_path,
        ], "analytics.py")
        if not ok:
            print("[FATAL] analytics.py falhou.")
 
        # Passo 3: Insights
        print("\n[3/4] Geração de insights com LLM")
        ok = run_module(INSIGHTS, [
            "--input",    metrics_path,
            "--output",   insights_path,
            "--strategy", "B",  # usar sempre few-shot para avaliação
        ], "insights.py")
        if not ok:
            print("[AVISO] insights.py falhou — métricas de anomalia não disponíveis.")
 
        # Passo 4: Report
        print("\n[4/4] Geração do relatório semanal")
        ok = run_module(
            SRC_DIR / "report.py", [
                "--input",   insights_path,
                "--metrics", metrics_path,
                "--output",  report_path,
            ], "report.py")
 
        # ── Avaliação
        print("Calcular métricas de avaliação")
 
        evaluation = {
            "meta": {
                "data_file":    args.data,
                "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline": {
                    "stitcher":  str(STITCHER),
                    "analytics": str(ANALYTICS),
                    "insights":  str(INSIGHTS),
                }
            },
            "stitching":          {},
            "anomaly_detection":  {},
            "hallucination":      {},
            "summary":            {},
        }
 
        # Métricas de stitching
        if Path(journeys_path).exists():
            print("\n  A avaliar qualidade do stitching")
            evaluation["stitching"] = evaluate_stitching(args.data, journeys_path)
        else:
            evaluation["stitching"] = {"error": "journeys.csv não foi produzido"}
 
        # Métricas de deteção de anomalias
        if Path(insights_path).exists() and Path(metrics_path).exists():
            print("  A avaliar deteção de anomalias")
            evaluation["anomaly_detection"] = evaluate_anomaly_detection(
                insights_path, metrics_path
            )
        else:
            evaluation["anomaly_detection"] = {"error": "insights.json ou metrics.json não disponíveis"}
 
        # Métricas de alucinação
        if Path(report_path).exists() and Path(metrics_path).exists():
            print("A avaliar alucinação no relatório")
            evaluation["hallucination"] = evaluate_hallucination(report_path, metrics_path)
        else:
            evaluation["hallucination"] = {"error": "weekly_report.md não disponível"}
 
        # Resumo
        s = evaluation["stitching"]
        a = evaluation["anomaly_detection"]
        h = evaluation["hallucination"]
        evaluation["summary"] = {
            "consistency":            s.get("consistency_pct", "N/A"),
            "coverage":               s.get("coverage_pct", "N/A"),
            "completeness":           s.get("completeness_pct", "N/A"),
            "anomalies_detected":     a.get("anomalies_in_metrics", "N/A"),
            "numeric_precision":      a.get("numeric_precision_pct", "N/A"),
            "hallucination_rate":     h.get("hallucination_pct", "N/A"),
            "total_time_s":           round(time.time() - t_total, 1),
        }
 
        # Escrever resultado
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
 
        # Imprimir resumo
        print("  Resultados:")
        for k, v in evaluation["summary"].items():
            print(f"  {k:<30} {v}")
        print(f"\n  Relatório completo em {args.output}")
 
 
if __name__ == "__main__":
    main()