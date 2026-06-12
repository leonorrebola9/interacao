import os
import json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
import time

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

REPORTS_DIR = "./data/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Prompt para o relatório
PROMPT_REPORT = """És um sistema de geração de relatórios de inspeção de prateleiras de supermercado.

Gera um relatório de inspeção em Markdown com as seguintes secções obrigatórias:

# Relatório de Inspeção — {date}

## 1. Sumário Executivo
Máximo 150 palavras. Estado geral da loja nesta sessão. Quantas zonas inspecionadas, quantos issues críticos, quantos warnings. Linguagem direta e acionável.

## 2. Problemas por Zona
Para cada zona com problemas: lista de problemas, severidade, fill rate, e comparação com histórico se disponível.

## 3. Regras Disparadas
Que regras foram ativadas, com que dados, e que ação foi gerada. Se não houver regras disparadas, indica isso.

## 4. Contexto Histórico Relevante
Padrões passados recuperados do histórico com referências explícitas (inspection_id, data). Se não houver histórico relevante, indica isso.

## 5. Recomendações
Máximo 5 ações concretas, ordenadas por urgência, cada uma específica o suficiente para ser executável sem interpretação adicional.

---

Dados da sessão de inspeção:
{session_data}

Contexto histórico recuperado:
{historical_context}

Notificações de regras disparadas:
{notifications}

Gera o relatório completo em Markdown. Sê direto e objetivo.
"""


# Relatório com:
    # inspections: lista de dicts de inspeção
    # notifications: lista de notificações geradas pelo rule_engine
    # historical_context: lista de inspeções recuperadas do RAG
def generate_report(inspections, notifications=None, historical_context=None, max_retries=3):
    if notifications is None:
        notifications = []
    if historical_context is None:
        historical_context = []

    # prepara dados da sessão
    session_data = {
        "total_inspections": len(inspections),
        "zones_inspected": list(set(i.get("zone_id", "") for i in inspections)),
        "critical_count": sum(1 for i in inspections if i.get("overall_status") == "critical"),
        "warning_count": sum(1 for i in inspections if i.get("overall_status") == "warning"),
        "ok_count": sum(1 for i in inspections if i.get("overall_status") == "ok"),
        "inspections": inspections
    }

    # prepara contexto histórico
    if historical_context:
        hist_text = "\n\n".join([
            f"[{h['inspection_id']}] {h['metadata'].get('date', '')} — "
            f"Zona: {h['metadata'].get('zone_id', '')} — "
            f"Status: {h['metadata'].get('overall_status', '')} — "
            f"Summary: {h['summary']}"
            for h in historical_context
        ])
    else:
        hist_text = "Sem histórico relevante disponível."

    # prepara notificações
    if notifications:
        notif_text = "\n\n".join([
            f"[{n['rule_id']}] Alert: {n['alert_level']} — {n['message']}"
            for n in notifications
        ])
    else:
        notif_text = "Nenhuma regra disparada nesta sessão."

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prompt = PROMPT_REPORT.format(
        date=date_str,
        session_data=json.dumps(session_data, ensure_ascii=False, indent=2),
        historical_context=hist_text,
        notifications=notif_text
    )

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

    raise RuntimeError("Limite de tentativas excedido")


# Guarda o relatório
def save_report(report_text, session_name=None):
    if session_name is None:
        session_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    filename = f"report_{session_name}.md"
    path = os.path.join(REPORTS_DIR, filename)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"Relatório guardado: {path}")
    return path


# Geração de relatório com contexto RAG e Rule Engine
def generate_full_report(inspections, rules=None, session_name=None):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rag_memory import query_memory, retrieve
    from rule_engine import execute_rules, load_rules

    if rules is None:
        rules = load_rules()

    # executa regras
    all_notifications = []
    all_logs = []
    for inspection in inspections:
        notifs, logs = execute_rules(inspection, rules)
        all_notifications.extend(notifs)
        all_logs.extend(logs)

    if all_logs:
        print("\nLogs de execução de regras:")
        for log in all_logs:
            print(f"  {log}")

    # recupera contexto histórico relevante
    zones = list(set(i.get("zone_id", "") for i in inspections))
    query = f"problemas nas zonas {', '.join(zones)}"
    _, historical = query_memory(query, k=3)

    # gera relatório
    print("\nA gerar relatório")
    report = generate_report(
        inspections=inspections,
        notifications=all_notifications,
        historical_context=historical
    )

    # guarda
    path = save_report(report, session_name)
    return report, path


#     Gera relatório dividido em dois grupos e concatena num único ficheiro.
def generate_split_report(inspections_dir="./data/inspections", session_name=None):
    all_inspections = []
    for path in Path(inspections_dir).glob("*.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        all_inspections.append(data)

    # grupo 1: fotos próprias estratégia B
    group1 = [i for i in all_inspections 
              if i.get("strategy") == "B" 
              and "sku_" not in str(i.get("image_path", ""))]

    # grupo 2: SKU estratégia A
    group2 = [i for i in all_inspections 
              if "sku_" in str(i.get("image_path", ""))]

    print(f"Grupo 1 (fotos próprias): {len(group1)} inspeções")
    print(f"Grupo 2 (SKU): {len(group2)} inspeções")

    print("\nA gerar relatório — Parte 1")
    report1, _ = generate_full_report(group1, session_name=f"{session_name}_part1")

    print("\nA gerar relatório — Parte 2")
    report2, _ = generate_full_report(group2, session_name=f"{session_name}_part2")

    # concatena
    if session_name is None:
        session_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    final_report = f"{report1}\n\n---\n\n{report2}"
    path = save_report(final_report, session_name)
    print(f"\nRelatório final guardado: {path}")
    return final_report, path

if __name__ == "__main__":
    import sys
    inspections = []

    if sys.argv[1] == "--split":
        folder = sys.argv[2] if len(sys.argv) > 2 else "./data/inspections"
        report, path = generate_split_report(folder)
        print(f"\nRelatório gerado: {path}")
        sys.exit(0)  # ← importante!

    elif sys.argv[1] == "--dir":
        folder = sys.argv[2] if len(sys.argv) > 2 else "./data/inspections"
        for path in Path(folder).glob("*.json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            inspections.append(data)
        print(f"Carregadas {len(inspections)} inspeções de {folder}")

    else:
        for path in sys.argv[1:]:
            with open(path, encoding="utf-8") as f:
                inspections.append(json.load(f))

    if not inspections:
        print("Nenhuma inspeção encontrada.")
        sys.exit(1)

    report, path = generate_full_report(inspections)
    print(f"\nRelatório gerado: {path}")
    print("\n--- PREVIEW ---")
    print(report[:500] + "...")