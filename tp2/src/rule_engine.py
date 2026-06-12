import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv
import time


# API
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"
RULES_DIR = "./data/rules"
os.makedirs(RULES_DIR, exist_ok=True)


# Prompt para conversão de regra
PROMPT_RULE_CONVERSION = """És um sistema de conversão de regras de inspeção de prateleiras de supermercado.

O gestor de loja vai escrever uma regra em linguagem natural. A tua tarefa é converter essa regra para JSON estruturado.

Schema obrigatório:
{
  "rule_id": "<gerado externamente>",
  "created_at": "<gerado externamente>",
  "natural_language": "<texto original da regra>",
  "description": "reformulação clara e inequívoca em português formal",
  "conditions": {
    "zone_filter": ["Z_S1", "Z_S3"],
    "time_filter": {"hours_start": 10, "hours_end": 13},
    "issue_types": ["empty_shelf", "damaged"],
    "severity_threshold": "low|medium|high",
    "fill_rate_threshold": 0.6,
    "location_filter": "bottom|middle|top|any"
  },
  "action": {
    "alert_level": "info|warning|critical",
    "notification_message": "template da mensagem quando a regra dispara"
  },
  "validation": {
    "is_valid": true,
    "ambiguities": ["lista de aspetos não claros"],
    "assumptions": ["lista de pressupostos assumidos na conversão"]
  }
}

Regras de conversão:
- Se zone_filter não for especificado, usa [] (significa todas as zonas)
- Se time_filter não for especificado, usa null
- Se issue_types não for especificado, usa [] (significa todos os tipos)
- Se severity_threshold não for especificado, usa "low" (apanha tudo)
- Se fill_rate_threshold não for especificado, usa null
- Se location_filter não for especificado, usa "any"
- alert_level: "info" para avisos não urgentes, "warning" para situações a monitorizar, "critical" para ação imediata
- notification_message deve ser um template com placeholders como {zone_id}, {fill_rate}, {issue_type}

Tipos de issue válidos: empty_shelf, wrong_product, damaged, misaligned, label_missing, other

DETEÇÃO DE AMBIGUIDADES — OBRIGATÓRIO:
Marca is_valid como false e lista em ambiguities SEMPRE que:
- "vazia" ou "empty" não especifica percentagem ou threshold numérico concreto
- não especifica a que zonas se aplica (zona específica vs todas as zonas)
- não especifica nível de urgência ou alert_level
- usa termos vagos como "muito", "pouco", "bastante", "suficiente" sem valor numérico
- não especifica se se aplica a uma prateleira específica (superior/meio/inferior) ou a todas

Exemplos de regras AMBÍGUAS que devem ter is_valid=false:
- "Avisa-me quando a prateleira estiver vazia" → ambígua: o que é "vazia"? que zonas? que urgência?
- "Notifica-me se houver problemas" → ambígua: que tipo de problemas? que severidade?
- "Alerta quando estiver muito vazio" → ambígua: quanto é "muito"?

Exemplos de regras CLARAS que podem ter is_valid=true:
- "Avisa-me quando o fill rate da zona Z_S1 cair abaixo de 60%" → clara: zona, threshold, ação definidos
- "Na zona Z_S2, se houver um wrong_product, nível crítico" → clara: zona, tipo, urgência definidos

Responde APENAS com o JSON. Sem texto adicional, sem markdown, sem ```json.
"""


# Conversão de regra de linguagem natural para JSON estruturado
def convert_rule(natural_language_text, max_retries=3): 
    prompt = f"{PROMPT_RULE_CONVERSION}\n\nRegra do gestor: \"{natural_language_text}\""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0)
            )
            raw_text = response.text.strip()

            import re
            matches = re.findall(r'\{[\s\S]*\}', raw_text)
            if not matches:
                raise ValueError("Nenhum JSON encontrado na resposta")

            data = json.loads(matches[-1])

            data["rule_id"] = f"RULE_{uuid.uuid4().hex[:6].upper()}"
            data["created_at"] = datetime.now(timezone.utc).isoformat()
            data["natural_language"] = natural_language_text

            return data

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [aviso] Erro a parsear JSON (tentativa {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise

        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                wait = 35 + attempt * 15
                print(f"  [aviso] Servidor indisponível, a aguardar {wait}s (tentativa {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Limite de tentativas excedido")


# Regras
# serve para guardar a regra
def save_rule(rule):
    """Guarda uma regra em disco."""
    path = os.path.join(RULES_DIR, f"{rule['rule_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rule, f, indent=2, ensure_ascii=False)
    return path

# Server para carregar as regras guardadas
def load_rules():
    rules = []
    for path in Path(RULES_DIR).glob("*.json"):
        with open(path, encoding="utf-8") as f:
            rules.append(json.load(f))
    return rules

# serve para carregar uma regra específica
def load_rule_by_id(rule_id):
    path = os.path.join(RULES_DIR, f"{rule_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# Serve para apagar uma regra
def delete_rule(rule_id):
    path = os.path.join(RULES_DIR, f"{rule_id}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

# Lista todas as regras com resumo
def list_rules():
    rules = load_rules()
    if not rules:
        print("Nenhuma regra guardada.")
        return []
    for r in rules:
        status = "✓" if r.get("validation", {}).get("is_valid") else "ambígua"
        print(f"  [{r['rule_id']}] {status} — {r.get('natural_language', '')[:60]}")
    return rules


# Execução de regras
# Verifica se uma regra dispara face aos resultados de uma inspeção
def check_rule(rule, inspection):
    conditions = rule.get("conditions", {})

    # filtro de zona
    zone_filter = conditions.get("zone_filter", [])
    if zone_filter and inspection.get("zone_id") not in zone_filter:
        return False, "zona não corresponde"

    # filtro de hora
    time_filter = conditions.get("time_filter")
    if time_filter:
        hour = datetime.now().hour
        if not (time_filter["hours_start"] <= hour < time_filter["hours_end"]):
            return False, "fora do horário definido"

    # filtro de fill_rate
    fill_threshold = conditions.get("fill_rate_threshold")
    if fill_threshold is not None:
        fill_rate = inspection.get("shelf_fill_rate", 1.0)
        if fill_rate >= fill_threshold:
            return False, f"fill_rate {fill_rate} acima do threshold {fill_threshold}"

    # filtro de issue_types e severity
    issue_types = conditions.get("issue_types", [])
    severity_threshold = conditions.get("severity_threshold", "low")
    location_filter = conditions.get("location_filter", "any")

    severity_order = {"low": 0, "medium": 1, "high": 2}
    min_severity = severity_order.get(severity_threshold, 0)

    issues = inspection.get("issues", [])

    matching_issues = []
    for issue in issues:
        # verifica tipo
        if issue_types and issue.get("type") not in issue_types:
            continue
        # verifica severidade
        issue_severity = severity_order.get(issue.get("severity", "low"), 0)
        if issue_severity < min_severity:
            continue
        # verifica localização
        if location_filter != "any":
            location = issue.get("location", "").lower()
            if location_filter not in location:
                continue
        matching_issues.append(issue)

    if not matching_issues:
        return False, "nenhum issue corresponde às condições"

    return True, matching_issues


# Executa todas as regras guardadas contra uma inspeção
# Retorna lista de notificações geradas
def execute_rules(inspection, rules=None):
    if rules is None:
        rules = load_rules()

    notifications = []
    logs = []

    for rule in rules:
        # só executa regras válidas
        if not rule.get("validation", {}).get("is_valid", True):
            logs.append(f"[SKIP] {rule['rule_id']} — regra inválida/ambígua")
            continue

        triggered, result = check_rule(rule, inspection)
        logs.append(f"[CHECK] {rule['rule_id']} — {'DISPAROU' if triggered else f'não disparou ({result})'}")

        if triggered:
            matching_issues = result
            template = rule.get("action", {}).get("notification_message", "Regra {rule_id} disparou na zona {zone_id}.")

            # preenche template
            message = template.format(
                rule_id=rule.get("rule_id", ""),
                zone_id=inspection.get("zone_id", ""),
                fill_rate=inspection.get("shelf_fill_rate", ""),
                issue_type=matching_issues[0].get("type", "") if matching_issues else "",
                severity=matching_issues[0].get("severity", "") if matching_issues else "",
                timestamp=inspection.get("timestamp", ""),
            )

            notifications.append({
                "rule_id": rule["rule_id"],
                "alert_level": rule.get("action", {}).get("alert_level", "info"),
                "message": message,
                "triggered_by": matching_issues,
                "inspection_id": inspection.get("inspection_id"),
                "zone_id": inspection.get("zone_id"),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    return notifications, logs


# Adição de regra
def add_rule_interactive(natural_language_text):
    print(f"\nA converter regra: \"{natural_language_text}\"")
    rule = convert_rule(natural_language_text)

    ambiguities = rule.get("validation", {}).get("ambiguities", [])
    assumptions = rule.get("validation", {}).get("assumptions", [])

    if assumptions:
        print("\nPressupostos assumidos:")
        for a in assumptions:
            print(f"  • {a}")

    if ambiguities:
        print("\nAmbiguidades detetadas:")
        for i, amb in enumerate(ambiguities, 1):
            print(f"  {i}. {amb}")

    path = save_rule(rule)
    print(f"\nRegra guardada: {rule['rule_id']}")
    print(f"Descrição: {rule.get('description', '')}")
    return rule


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso:")
        print("  python src/rule_engine.py add \"<regra em linguagem natural>\"")
        print("  python src/rule_engine.py list")
        print("  python src/rule_engine.py delete <RULE_ID>")
        print("  python src/rule_engine.py test <RULE_ID> <inspection.json>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "add":
        if len(sys.argv) < 3:
            print("Erro: falta a regra.")
            sys.exit(1)
        rule_text = sys.argv[2]
        add_rule_interactive(rule_text)

    elif command == "list":
        list_rules()

    elif command == "delete":
        if len(sys.argv) < 3:
            print("Erro: falta o RULE_ID.")
            sys.exit(1)
        rule_id = sys.argv[2]
        if delete_rule(rule_id):
            print(f"Regra {rule_id} apagada.")
        else:
            print(f"Regra {rule_id} não encontrada.")

    elif command == "test":
        if len(sys.argv) < 4:
            print("Erro: falta RULE_ID e/ou inspection.json")
            sys.exit(1)
        rule_id = sys.argv[2]
        inspection_path = sys.argv[3]

        rule = load_rule_by_id(rule_id)
        if not rule:
            print(f"Regra {rule_id} não encontrada.")
            sys.exit(1)

        with open(inspection_path, encoding="utf-8") as f:
            inspection = json.load(f)

        triggered, result = check_rule(rule, inspection)
        if triggered:
            print(f"Regra disparou — {len(result)} issue(s) correspondente(s)")
            for issue in result:
                print(f"  • [{issue['severity']}] {issue['type']} — {issue['location']}")
        else:
            print(f"Regra não disparou — {result}")
    else:
        print(f"Comando desconhecido: {command}")