import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from shelf_inspector import inspect_image

VALID_ISSUE_TYPES = ["empty_shelf", "wrong_product", "damaged", "misaligned", "label_missing", "other"]
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
ANNOTATIONS_DIR = "./data/annotations"

# ─────────────────────────────────────────────
# CARREGAMENTO DE ANOTACOES
# ─────────────────────────────────────────────

def load_annotation(image_path):
    """Carrega a anotacao ground truth para uma imagem."""
    image_name = Path(image_path).stem
    annotation_path = os.path.join(ANNOTATIONS_DIR, f"{image_name}.json")
    if not os.path.exists(annotation_path):
        return None
    with open(annotation_path, encoding="utf-8") as f:
        data = json.load(f)
    # normaliza overall_status
    status = data.get("overall_status", "ok").lower()
    if status not in ["ok", "warning", "critical"]:
        status = "ok"
    data["overall_status"] = status
    return data


# ─────────────────────────────────────────────
# METRICAS DE INSPECAO VISUAL
# ─────────────────────────────────────────────

def evaluate_inspection(prediction, ground_truth):
    """
    Avalia uma inspecao face ao ground truth.
    Retorna dict com metricas.
    """
    pred_issues = prediction.get("issues", [])
    gt_issues = ground_truth.get("issues", [])

    pred_types = [i.get("type") for i in pred_issues]
    gt_types = [i.get("type") for i in gt_issues]

    # Issue Detection Rate (Recall) — % de issues do GT corretamente identificados
    detected = sum(1 for gt in gt_types if gt in pred_types)
    issue_detection_rate = detected / len(gt_types) if gt_types else 1.0

    # False Positive Rate — % de issues reportados que nao existem no GT
    false_positives = sum(1 for p in pred_types if p not in gt_types)
    false_positive_rate = false_positives / len(pred_types) if pred_types else 0.0

    # Severity Accuracy — % de issues com severidade correta
    severity_correct = 0
    severity_total = 0
    for gt_issue in gt_issues:
        gt_type = gt_issue.get("type")
        gt_sev = gt_issue.get("severity", "low")
        # procura issue correspondente na predicao
        matching = [p for p in pred_issues if p.get("type") == gt_type]
        if matching:
            pred_sev = matching[0].get("severity", "low")
            if pred_sev == gt_sev:
                severity_correct += 1
            severity_total += 1

    severity_accuracy = severity_correct / severity_total if severity_total > 0 else None

    # JSON Parse Rate — se chegamos aqui, o JSON foi parseado com sucesso
    json_parse_rate = 1.0

    return {
        "issue_detection_rate": round(issue_detection_rate, 3),
        "false_positive_rate": round(false_positive_rate, 3),
        "severity_accuracy": round(severity_accuracy, 3) if severity_accuracy is not None else None,
        "json_parse_rate": json_parse_rate,
        "gt_issues": gt_types,
        "pred_issues": pred_types,
        "gt_status": ground_truth.get("overall_status"),
        "pred_status": prediction.get("overall_status")
    }


# ─────────────────────────────────────────────
# AVALIACAO POR ESTRATEGIA
# ─────────────────────────────────────────────

def evaluate_strategy(images, strategy, zone_id="Z_EVAL", delay=6):
    """
    Avalia uma estrategia de prompting sobre um conjunto de imagens anotadas.
    """
    results = []
    json_failures = 0

    print(f"\nA avaliar estrategia {strategy} em {len(images)} imagens...")

    for i, image_path in enumerate(images):
        annotation = load_annotation(image_path)
        if not annotation:
            print(f"  [{i+1}/{len(images)}] {Path(image_path).name} — sem anotacao, a saltar")
            continue

        print(f"  [{i+1}/{len(images)}] {Path(image_path).name}...", end=" ")

        try:
            prediction = inspect_image(image_path, zone_id=zone_id, strategy=strategy)
            metrics = evaluate_inspection(prediction, annotation)
            metrics["image"] = Path(image_path).name
            metrics["strategy"] = strategy
            results.append(metrics)
            print(f"IDR: {metrics['issue_detection_rate']} | FPR: {metrics['false_positive_rate']}")

        except Exception as e:
            json_failures += 1
            print(f"ERRO: {e}")

        time.sleep(delay)

    # agrega metricas
    if not results:
        return {}

    avg_idr = sum(r["issue_detection_rate"] for r in results) / len(results)
    avg_fpr = sum(r["false_positive_rate"] for r in results) / len(results)

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


# ─────────────────────────────────────────────
# AVALIACAO DO RAG
# ─────────────────────────────────────────────

def evaluate_rag(queries_with_ground_truth, k=3):
    """
    Avalia o RAG com Recall@k.
    queries_with_ground_truth: lista de {query, relevant_inspection_ids}
    """
    from rag_memory import retrieve

    recall_hits = 0

    print(f"\nA avaliar RAG com {len(queries_with_ground_truth)} queries...")

    for item in queries_with_ground_truth:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        retrieved = retrieve(query, k=k)
        retrieved_ids = [r["inspection_id"] for r in retrieved]

        hit = any(rid in retrieved_ids for rid in relevant_ids)
        if hit:
            recall_hits += 1
        print(f"  Query: \"{query[:50]}\" — {'HIT' if hit else 'MISS'}")

    recall_at_k = recall_hits / len(queries_with_ground_truth) if queries_with_ground_truth else 0.0

    return {
        "recall_at_k": round(recall_at_k, 3),
        "k": k,
        "queries_evaluated": len(queries_with_ground_truth)
    }


# ─────────────────────────────────────────────
# LLM-AS-JUDGE
# ─────────────────────────────────────────────

def llm_judge(prediction_text, criterion, max_retries=3):
    """
    Usa o Gemini como juiz para avaliacao qualitativa.
    Retorna score (0-5) e justificacao.
    """
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
    from dotenv import load_dotenv
    load_dotenv()

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""És um avaliador de sistemas de inspeção de prateleiras de supermercado.

Avalia o seguinte output do sistema com base no critério fornecido.

Critério de avaliação: {criterion}

Output do sistema:
{prediction_text}

Responde APENAS com este formato:
score: <0 a 5>
justificacao: <uma frase curta explicando o score>
"""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0)
            )
            text = response.text.strip()
            score, justificacao = 0, ""
            for line in text.splitlines():
                if line.startswith("score:"):
                    try:
                        score = int(line.split(":")[1].strip())
                    except:
                        pass
                elif line.startswith("justificacao:"):
                    justificacao = line.split(":", 1)[1].strip()
            return {"score": score, "justificacao": justificacao}

        except ClientError as e:
            if e.code in [429, 503]:
                wait = 35 + attempt * 15
                print(f"  [aviso] Erro {e.code}, a aguardar {wait}s...")
                time.sleep(wait)
            else:
                raise

    return {"score": 0, "justificacao": "Erro na avaliacao"}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Harness de avaliacao do sistema")
    parser.add_argument("--images-dir", required=True, help="Pasta com imagens de teste")
    parser.add_argument("--output", default="evaluation_report.json", help="Ficheiro de output")
    parser.add_argument("--strategies", default="A,B,C", help="Estrategias a avaliar (A,B,C)")
    parser.add_argument("--zone", default="Z_EVAL", help="Zona para as inspecoes")
    parser.add_argument("--delay", type=int, default=6, help="Delay entre chamadas API (segundos)")
    args = parser.parse_args()

    # recolhe imagens com anotacao
    images_dir = args.images_dir
    all_images = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    annotated_images = [str(img) for img in all_images if load_annotation(str(img)) is not None]

    print(f"Imagens encontradas: {len(all_images)}")
    print(f"Imagens com anotacao: {len(annotated_images)}")

    if not annotated_images:
        print("Nenhuma imagem anotada encontrada. Verifica a pasta de anotacoes.")
        return

    strategies = [s.strip() for s in args.strategies.split(",")]
    evaluation_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "images_dir": images_dir,
        "total_images": len(annotated_images),
        "strategies": {}
    }

    # avalia cada estrategia
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"ESTRATEGIA {strategy}")
        print(f"{'='*50}")
        result = evaluate_strategy(annotated_images, strategy, zone_id=args.zone, delay=args.delay)
        evaluation_results["strategies"][strategy] = result

    # resumo comparativo
    print(f"\n{'='*50}")
    print("RESUMO COMPARATIVO")
    print(f"{'='*50}")
    print(f"{'Metrica':<25} {'A':>8} {'B':>8} {'C':>8}")
    print("-" * 50)

    metrics = ["issue_detection_rate", "false_positive_rate", "severity_accuracy", "json_parse_rate"]
    labels = ["Issue Detection Rate", "False Positive Rate", "Severity Accuracy", "JSON Parse Rate"]

    for metric, label in zip(metrics, labels):
        values = []
        for s in strategies:
            v = evaluation_results["strategies"].get(s, {}).get(metric)
            values.append(f"{v:.3f}" if v is not None else "N/A")
        print(f"{label:<25} {values[0]:>8} {values[1] if len(values) > 1 else 'N/A':>8} {values[2] if len(values) > 2 else 'N/A':>8}")

    # guarda resultados
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados em: {args.output}")


if __name__ == "__main__":
    main()