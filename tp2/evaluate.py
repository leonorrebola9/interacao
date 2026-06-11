import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from shelf_inspector import inspect_image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.0-flash"

VALID_ISSUE_TYPES = ["empty_shelf", "wrong_product", "damaged", "misaligned", "label_missing", "other"]
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
GROUND_TRUTH_PATH = "./data/annotations/ground_truth.json"

with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
    GROUND_TRUTH = json.load(f)


# ─── GROUND TRUTH ─────────────────────────────────────────────────────────────

def load_annotation(image_path):
    image_name = Path(image_path).name
    entry = GROUND_TRUTH.get(image_name)
    if not entry:
        return None
    status = entry.get("overall_status", "ok").lower()
    if status not in ["ok", "warning", "critical"]:
        status = "ok"
    return {
        "overall_status": status,
        "fill_rate": entry.get("shelf_fill_rate", 1.0),
        "issues": entry.get("issues", []),
        "zone": entry.get("zone", "Z_S1")
    }

def get_zone(image_path):
    image_name = Path(image_path).name
    entry = GROUND_TRUTH.get(image_name)
    return entry.get("zone", "Z_S1") if entry else "Z_S1"


# ─── MÉTRICAS ANÁLISE VISUAL ──────────────────────────────────────────────────

def evaluate_inspection(prediction, ground_truth):
    pred_issues = prediction.get("issues", [])
    gt_issues = ground_truth.get("issues", [])
    pred_types = [i.get("type") for i in pred_issues]
    gt_types = [i.get("type") for i in gt_issues]

    if gt_types:
        detected = sum(1 for gt in gt_types if gt in pred_types)
        issue_detection_rate = detected / len(gt_types)
    else:
        issue_detection_rate = None

    if pred_types:
        false_positives = sum(1 for p in pred_types if p not in gt_types)
        false_positive_rate = false_positives / len(pred_types)
    else:
        false_positive_rate = 0.0

    severity_correct = 0
    severity_total = 0
    for gt_issue in gt_issues:
        gt_type = gt_issue.get("type")
        gt_sev = gt_issue.get("severity", "low")
        matching = [p for p in pred_issues if p.get("type") == gt_type]
        if matching:
            pred_sev = matching[0].get("severity", "low")
            if pred_sev == gt_sev:
                severity_correct += 1
            severity_total += 1

    severity_accuracy = severity_correct / severity_total if severity_total > 0 else None

    return {
        "issue_detection_rate": round(issue_detection_rate, 3) if issue_detection_rate is not None else None,
        "false_positive_rate": round(false_positive_rate, 3),
        "severity_accuracy": round(severity_accuracy, 3) if severity_accuracy is not None else None,
        "json_parse_rate": 1.0,
        "gt_issues": gt_types,
        "pred_issues": pred_types,
        "gt_status": ground_truth.get("overall_status"),
        "pred_status": prediction.get("overall_status")
    }


def evaluate_strategy(images, strategy, delay=6):
    results = []
    json_failures = 0

    print(f"\nA avaliar estratégia {strategy} em {len(images)} imagens")

    for i, image_path in enumerate(images):
        annotation = load_annotation(image_path)
        if not annotation:
            print(f"  [{i+1}/{len(images)}] {Path(image_path).name} — sem anotação, a saltar")
            continue

        print(f"  [{i+1}/{len(images)}] {Path(image_path).name}...", end=" ")

        try:
            zone = get_zone(image_path)
            prediction = inspect_image(image_path, zone_id=zone, strategy=strategy)
            metrics = evaluate_inspection(prediction, annotation)
            metrics["image"] = Path(image_path).name
            metrics["strategy"] = strategy
            metrics["zone"] = zone
            results.append(metrics)
            print(f"IDR: {metrics['issue_detection_rate']} | FPR: {metrics['false_positive_rate']}")
        except Exception as e:
            json_failures += 1
            print(f"ERRO: {e}")

        time.sleep(delay)

    if not results:
        return {}

    idr_results = [r["issue_detection_rate"] for r in results if r["issue_detection_rate"] is not None]
    avg_idr = sum(idr_results) / len(idr_results) if idr_results else 0.0
    fpr_results = [r["false_positive_rate"] for r in results]
    avg_fpr = sum(fpr_results) / len(fpr_results) if fpr_results else 0.0
    sev_results = [r["severity_accuracy"] for r in results if r["severity_accuracy"] is not None]
    avg_sev = sum(sev_results) / len(sev_results) if sev_results else None
    total_attempts = len(results) + json_failures
    json_parse_rate = len(results) / total_attempts if total_attempts > 0 else 0.0

    return {
        "strategy": strategy,
        "images_evaluated": len(results),
        "issue_detection_rate": round(avg_idr, 3),
        "false_positive_rate": round(avg_fpr, 3),
        "severity_accuracy": round(avg_sev, 3) if avg_sev is not None else None,
        "json_parse_rate": round(json_parse_rate, 3),
        "per_image": results
    }


# ─── MÉTRICAS RAG ─────────────────────────────────────────────────────────────

# queries com ground truth definido manualmente
RAG_QUERIES = [
    {
        "query": "prateleira vazia zona Z_S1",
        "relevant_ids": ["INS_20260610_173806_3A2E5E", "INS_20260610_165208_23E58F"]
    },
    {
        "query": "produto fora do lugar cereais wrong product",
        "relevant_ids": ["INS_20260610_161601_44624E", "INS_20260610_161951_188387"]
    },
    {
        "query": "prateleira vazia zona Z_S2 padaria",
        "relevant_ids": ["INS_20260610_165736_6B9B6C", "INS_20260610_174914_CE3289"]
    },
    {
        "query": "fill rate baixo zona Z_S6 vinhos destilados",
        "relevant_ids": ["INS_20260610_161601_44624E", "INS_20260610_174851_8F62EB"]
    },
    {
        "query": "produto tombado misaligned detergentes limpeza",
        "relevant_ids": ["INS_20260610_175625_1AD94A", "INS_20260610_175145_08E92A"]
    }
]

def evaluate_rag(k=3):
    from rag_memory import retrieve

    recall_hits = 0
    print(f"\nA avaliar RAG com {len(RAG_QUERIES)} queries (Recall@{k})")

    results = []
    for item in RAG_QUERIES:
        query = item["query"]
        relevant_ids = item["relevant_ids"]
        retrieved = retrieve(query, k=k)
        retrieved_ids = [r["inspection_id"] for r in retrieved]
        hit = any(rid in retrieved_ids for rid in relevant_ids)
        if hit:
            recall_hits += 1
        print(f"  Query: \"{query[:50]}\" — {'HIT ✓' if hit else 'MISS ✗'}")
        results.append({
            "query": query,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "hit": hit
        })

    recall_at_k = recall_hits / len(RAG_QUERIES) if RAG_QUERIES else 0.0
    return {
        "recall_at_k": round(recall_at_k, 3),
        "k": k,
        "queries_evaluated": len(RAG_QUERIES),
        "hits": recall_hits,
        "per_query": results
    }


# ─── MÉTRICAS RULE ENGINE ─────────────────────────────────────────────────────

# regras de teste com dados sintéticos
RULE_TESTS = [
    {
        "rule_text": "Na zona Z_S1, se o fill rate cair abaixo de 60%, avisa com nível crítico",
        "synthetic_inspection": {
            "zone_id": "Z_S1",
            "overall_status": "critical",
            "shelf_fill_rate": 0.45,
            "issues": [{"type": "empty_shelf", "severity": "high", "location": "prateleira inferior"}],
            "inspection_id": "TEST_001",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "should_trigger": True
    },
    {
        "rule_text": "Se houver um produto fora do lugar em qualquer zona, nível de alerta warning",
        "synthetic_inspection": {
            "zone_id": "Z_S5",
            "overall_status": "warning",
            "shelf_fill_rate": 0.90,
            "issues": [{"type": "wrong_product", "severity": "medium", "location": "prateleira central"}],
            "inspection_id": "TEST_002",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "should_trigger": True
    },
    {
        "rule_text": "Se o fill rate de qualquer zona cair abaixo de 70%, avisa com nível warning",
        "synthetic_inspection": {
            "zone_id": "Z_S3",
            "overall_status": "ok",
            "shelf_fill_rate": 0.95,
            "issues": [],
            "inspection_id": "TEST_003",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "should_trigger": False
    },
    {
        "rule_text": "Avisa-me quando a prateleira estiver vazia",
        "expected_ambiguous": True
    },
    {
        "rule_text": "Notifica-me se a secção de padaria tiver problemas",
        "expected_ambiguous": True
    }
]

def evaluate_rule_engine(delay=4):
    from rule_engine import convert_rule, check_rule

    parse_successes = 0
    correctness_correct = 0
    correctness_total = 0
    ambiguity_detected = 0
    ambiguity_total = 0

    print(f"\nA avaliar Rule Engine com {len(RULE_TESTS)} testes")

    results = []
    for i, test in enumerate(RULE_TESTS):
        rule_text = test["rule_text"]
        print(f"  [{i+1}/{len(RULE_TESTS)}] \"{rule_text[:50]}\"...", end=" ")

        try:
            rule = convert_rule(rule_text)
            parse_successes += 1

            is_ambiguous = not rule.get("validation", {}).get("is_valid", True)
            expected_ambiguous = test.get("expected_ambiguous", False)

            if expected_ambiguous:
                ambiguity_total += 1
                if is_ambiguous:
                    ambiguity_detected += 1
                    print("ambígua ✓")
                else:
                    print("ambígua ✗ (não detetada)")
            elif "synthetic_inspection" in test:
                should_trigger = test["should_trigger"]
                triggered, _ = check_rule(rule, test["synthetic_inspection"])
                correctness_total += 1
                if triggered == should_trigger:
                    correctness_correct += 1
                    print(f"correcta ✓ (trigger={triggered})")
                else:
                    print(f"incorrecta ✗ (esperado={should_trigger}, obtido={triggered})")

            results.append({
                "rule_text": rule_text,
                "parse_success": True,
                "is_ambiguous": is_ambiguous,
                "expected_ambiguous": expected_ambiguous
            })

        except Exception as e:
            print(f"ERRO: {e}")
            results.append({"rule_text": rule_text, "parse_success": False, "error": str(e)})

        time.sleep(delay)

    rule_parse_rate = parse_successes / len(RULE_TESTS) if RULE_TESTS else 0.0
    rule_correctness = correctness_correct / correctness_total if correctness_total > 0 else None
    ambiguity_detection = ambiguity_detected / ambiguity_total if ambiguity_total > 0 else None

    return {
        "rule_parse_rate": round(rule_parse_rate, 3),
        "rule_correctness": round(rule_correctness, 3) if rule_correctness is not None else None,
        "ambiguity_detection": round(ambiguity_detection, 3) if ambiguity_detection is not None else None,
        "tests_run": len(RULE_TESTS),
        "per_test": results
    }


# ─── LLM-AS-JUDGE ─────────────────────────────────────────────────────────────

def llm_judge(prediction_text, criterion, max_retries=3):
    prompt = f"""És um avaliador de sistemas de inspeção de prateleiras de supermercado.

Avalia o seguinte output do sistema com base no critério fornecido.

Critério de avaliação: {criterion}

Output do sistema:
{prediction_text}

Responde APENAS com este formato JSON:
{{"score": <0 a 5>, "justificacao": "<uma frase curta explicando o score>"}}

Sem texto adicional, sem markdown."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0)
            )
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            data = json.loads(text)
            return {"score": data.get("score", 0), "justificacao": data.get("justificacao", "")}

        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                wait = 35 + attempt * 15
                print(f"  [aviso] Servidor indisponível, a aguardar {wait}s...")
                time.sleep(wait)
            else:
                return {"score": 0, "justificacao": f"Erro: {str(e)}"}

    return {"score": 0, "justificacao": "Limite de tentativas excedido"}


def evaluate_reports_with_judge(reports_dir="./data/reports"):
    """Avalia relatórios gerados com LLM-as-Judge."""
    criteria = [
        "O relatório é claro, direto e acionável para um gestor de loja?",
        "O relatório cita corretamente os inspection_ids e datas?",
        "As recomendações são específicas e ordenadas por urgência?"
    ]

    report_files = list(Path(reports_dir).glob("*.md"))
    if not report_files:
        print("Nenhum relatório encontrado.")
        return {}

    # avalia o relatório mais recente
    report_path = sorted(report_files)[-1]
    print(f"\nA avaliar relatório: {report_path.name}")

    with open(report_path, encoding="utf-8") as f:
        report_text = f.read()[:3000]  # primeiros 3000 chars

    scores = []
    for criterion in criteria:
        print(f"  Critério: \"{criterion[:50]}\"...", end=" ")
        result = llm_judge(report_text, criterion)
        scores.append(result["score"])
        print(f"Score: {result['score']}/5 — {result['justificacao']}")
        time.sleep(4)

    avg_score = sum(scores) / len(scores) if scores else 0
    return {
        "report_evaluated": report_path.name,
        "average_score": round(avg_score, 2),
        "max_score": 5,
        "criteria_scores": scores
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Harness de avaliação do sistema")
    parser.add_argument("--images-dir", default="./data/images", help="Pasta com imagens de teste")
    parser.add_argument("--output", default="evaluation_report.json", help="Ficheiro de output")
    parser.add_argument("--strategies", default="A,B,C", help="Estratégias a avaliar (A,B,C)")
    parser.add_argument("--delay", type=int, default=6, help="Delay entre chamadas API")
    parser.add_argument("--skip-visual", action="store_true", help="Salta avaliação visual")
    parser.add_argument("--skip-rag", action="store_true", help="Salta avaliação RAG")
    parser.add_argument("--skip-rules", action="store_true", help="Salta avaliação Rule Engine")
    parser.add_argument("--skip-judge", action="store_true", help="Salta LLM-as-Judge")
    args = parser.parse_args()

    evaluation_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "images_dir": args.images_dir,
    }

    # ── avaliação visual ──
    if not args.skip_visual:
        all_images = list(Path(args.images_dir).glob("*.jpg")) + list(Path(args.images_dir).glob("*.png"))
        annotated_images = [str(img) for img in all_images if load_annotation(str(img)) is not None]

        print(f"\nImagens encontradas: {len(all_images)}")
        print(f"Imagens com anotação: {len(annotated_images)}")

        if annotated_images:
            strategies = [s.strip() for s in args.strategies.split(",")]
            evaluation_results["visual"] = {}

            for strategy in strategies:
                print(f"\n{'='*50}")
                print(f"ESTRATÉGIA {strategy}")
                print(f"{'='*50}")
                result = evaluate_strategy(annotated_images, strategy, delay=args.delay)
                evaluation_results["visual"][strategy] = result

            print(f"\n{'='*50}")
            print("RESUMO COMPARATIVO — ANÁLISE VISUAL")
            print(f"{'='*50}")
            print(f"{'Métrica':<25} {'A':>8} {'B':>8} {'C':>8}")
            print("-" * 50)

            metrics = ["issue_detection_rate", "false_positive_rate", "severity_accuracy", "json_parse_rate"]
            labels = ["Issue Detection Rate", "False Positive Rate", "Severity Accuracy", "JSON Parse Rate"]

            for metric, label in zip(metrics, labels):
                values = []
                for s in strategies:
                    v = evaluation_results["visual"].get(s, {}).get(metric)
                    values.append(f"{v:.3f}" if v is not None else " N/A")
                print(f"{label:<25} {values[0]:>8} {values[1] if len(values) > 1 else 'N/A':>8} {values[2] if len(values) > 2 else 'N/A':>8}")

    # ── avaliação RAG ──
    if not args.skip_rag:
        print(f"\n{'='*50}")
        print("AVALIAÇÃO RAG")
        print(f"{'='*50}")
        rag_results = evaluate_rag(k=3)
        evaluation_results["rag"] = rag_results
        print(f"\nRecall@3: {rag_results['recall_at_k']} ({rag_results['hits']}/{rag_results['queries_evaluated']} queries)")

    # ── avaliação rule engine ──
    if not args.skip_rules:
        print(f"\n{'='*50}")
        print("AVALIAÇÃO RULE ENGINE")
        print(f"{'='*50}")
        rule_results = evaluate_rule_engine(delay=args.delay)
        evaluation_results["rule_engine"] = rule_results
        print(f"\nRule Parse Rate: {rule_results['rule_parse_rate']}")
        print(f"Rule Correctness: {rule_results['rule_correctness']}")
        print(f"Ambiguity Detection: {rule_results['ambiguity_detection']}")

    # ── llm-as-judge ──
    if not args.skip_judge:
        print(f"\n{'='*50}")
        print("LLM-AS-JUDGE")
        print(f"{'='*50}")
        judge_results = evaluate_reports_with_judge()
        evaluation_results["llm_judge"] = judge_results
        if judge_results:
            print(f"\nScore médio: {judge_results['average_score']}/5")

    # guarda resultados
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Resultados guardados em: {args.output}")


if __name__ == "__main__":
    main()