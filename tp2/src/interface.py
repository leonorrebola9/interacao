import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# se estiver em src/, importa directamente
from shelf_inspector import inspect_image, inspect_directory
from rule_engine import (
    add_rule_interactive, list_rules, delete_rule,
    load_rule_by_id, check_rule, load_rules
)
from rag_memory import query_memory, index_inspection, get_stats
from report_generator import generate_full_report

HELP_TEXT = """
Comandos disponíveis:

  INSPECAO:
    inspect <imagem.jpg> [zona] [estrategia]     — inspeciona uma imagem
    inspect-dir <pasta> [zona] [estrategia]      — inspeciona uma pasta de imagens

  REGRAS:
    add rule "<regra em portugues>"              — adiciona uma nova regra
    list rules                                   — lista todas as regras
    delete rule <RULE_ID>                        — apaga uma regra
    test rule <RULE_ID> <imagem.jpg>             — testa uma regra numa imagem

  HISTORICO:
    history "<pergunta>"                         — consulta o historico de inspecoes
    history "<pergunta>" --zone <zona>           — filtra por zona
    stats                                        — mostra estatisticas do historico

  RELATORIO:
    report <inspecao.json> [inspecao2.json ...]  — gera relatorio de inspecoes
    report --dir <pasta>                         — gera relatorio de pasta de inspecoes

  GERAL:
    help                                         — mostra esta mensagem
    exit                                         — sai do sistema
"""


# Parseia uma linha de comando
def parse_command(line):
    line = line.strip()
    if not line:
        return None, []

    # comandos com aspas (ex: add rule "...")
    import shlex
    try:
        parts = shlex.split(line)
    except ValueError:
        parts = line.split()

    return parts[0].lower(), parts[1:]


def handle_inspect(args):
    if not args:
        print("Erro: falta o caminho da imagem.")
        return

    image_path = args[0]
    zone = args[1] if len(args) > 1 else "Z_S1"
    strategy = args[2] if len(args) > 2 else "A"

    if not os.path.exists(image_path):
        print(f"Erro: ficheiro nao encontrado: {image_path}")
        return

    print(f"A inspecionar {image_path} | Zona: {zone} | Estrategia: {strategy}")
    result = inspect_image(image_path, zone_id=zone, strategy=strategy)

    print(f"\nResultado:")
    print(f"  Status: {result.get('overall_status')}")
    print(f"  Fill rate: {result.get('shelf_fill_rate')}")
    print(f"  Issues: {len(result.get('issues', []))}")

    for issue in result.get("issues", []):
        print(f"    [{issue.get('severity')}] {issue.get('type')} — {issue.get('location')}")

    # indexa automaticamente no RAG
    print("\nA indexar no historico")
    index_inspection(result)

    # executa regras
    rules = load_rules()
    if rules:
        from rule_engine import execute_rules
        notifications, logs = execute_rules(result, rules)
        if notifications:
            print("\nAlertas:")
            for n in notifications:
                print(f"  [{n['alert_level'].upper()}] {n['message']}")

    print(f"\nInspecao concluida: {result.get('inspection_id')}")
    return result


def handle_inspect_dir(args):
    if not args:
        print("Erro: falta o caminho da pasta.")
        return

    folder = args[0]
    zone = args[1] if len(args) > 1 else "Z_S1"
    strategy = args[2] if len(args) > 2 else "A"

    if not os.path.exists(folder):
        print(f"Erro: pasta nao encontrada: {folder}")
        return

    results = inspect_directory(folder, zone_id=zone, strategy=strategy)
    print(f"\n{len(results)} imagens inspecionadas.")

    for result in results:
        print("\nA indexar no historico")
        index_inspection(result)

    return results


def handle_add_rule(args):
    if not args:
        print("Erro: falta a regra.")
        return
    rule_text = " ".join(args)
    add_rule_interactive(rule_text)


def handle_list_rules(args):
    rules = list_rules()
    if not rules:
        print("Nenhuma regra guardada.")


def handle_delete_rule(args):
    if not args:
        print("Erro: falta o RULE_ID.")
        return
    rule_id = args[0]
    if delete_rule(rule_id):
        print(f"Regra {rule_id} apagada.")
    else:
        print(f"Regra {rule_id} nao encontrada.")


def handle_test_rule(args):
    if len(args) < 2:
        print("Erro: uso — test rule <RULE_ID> <imagem.jpg>")
        return

    rule_id = args[0]
    image_path = args[1]
    strategy = args[2] if len(args) > 2 else "A"

    rule = load_rule_by_id(rule_id)
    if not rule:
        print(f"Regra {rule_id} nao encontrada.")
        return

    if not os.path.exists(image_path):
        print(f"Ficheiro nao encontrado: {image_path}")
        return

    print(f"A inspecionar {image_path} para testar regra {rule_id}...")
    result = inspect_image(image_path, strategy=strategy)

    triggered, outcome = check_rule(rule, result)
    if triggered:
        print(f"Regra DISPAROU — {len(outcome)} issue(s) correspondente(s):")
        for issue in outcome:
            print(f"  [{issue['severity']}] {issue['type']} — {issue['location']}")
    else:
        print(f"Regra nao disparou — {outcome}")


def handle_history(args):
    if not args:
        print("Erro: falta a pergunta.")
        return

    zone = None
    query_parts = []

    i = 0
    while i < len(args):
        if args[i] == "--zone" and i + 1 < len(args):
            zone = args[i + 1]
            i += 2
        else:
            query_parts.append(args[i])
            i += 1

    query = " ".join(query_parts)
    print(f"A consultar historico: \"{query}\"")

    answer, sources = query_memory(query, k=3, zone_filter=zone)
    print(f"\nResposta:\n{answer}")

    if sources:
        print(f"\nFontes ({len(sources)}):")
        for s in sources:
            meta = s["metadata"]
            print(f"  [{s['inspection_id']}] {meta.get('date', '')} — Zona: {meta.get('zone_id', '')} — Similaridade: {s['similarity']}")


def handle_stats(args):
    get_stats()


def handle_report(args):
    if not args:
        print("Erro: falta o ficheiro de inspecao ou --dir <pasta>.")
        return

    inspections = []

    if args[0] == "--dir":
        folder = args[1] if len(args) > 1 else "./data/inspections"
        for path in Path(folder).glob("*.json"):
            with open(path, encoding="utf-8") as f:
                inspections.append(json.load(f))
        print(f"Carregadas {len(inspections)} inspecoes.")
    else:
        for path in args:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    inspections.append(json.load(f))
            else:
                print(f"Ficheiro nao encontrado: {path}")

    if not inspections:
        print("Nenhuma inspecao encontrada.")
        return

    report, path = generate_full_report(inspections)
    print(f"\nRelatorio gerado: {path}")



def main():
    print("Retail Vision Intelligence System")
    print("Escreve 'help' para ver os comandos disponiveis.")
    print()

    session_results = []

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nA sair...")
            break

        if not line:
            continue

        cmd, args = parse_command(line)

        if cmd in ("exit", "quit", "sair"):
            print("A sair")
            break

        elif cmd == "help":
            print(HELP_TEXT)

        elif cmd == "inspect":
            if args and args[0] == "--dir":
                handle_inspect_dir(args[1:])
            else:
                result = handle_inspect(args)
                if result:
                    session_results.append(result)

        elif cmd == "inspect-dir":
            handle_inspect_dir(args)

        elif cmd == "add" and args and args[0] == "rule":
            handle_add_rule(args[1:])

        elif cmd == "list" and args and args[0] == "rules":
            handle_list_rules(args)

        elif cmd == "delete" and args and args[0] == "rule":
            handle_delete_rule(args[1:])

        elif cmd == "test" and args and args[0] == "rule":
            handle_test_rule(args[1:])

        elif cmd == "history":
            handle_history(args)

        elif cmd == "stats":
            handle_stats(args)

        elif cmd == "report":
            handle_report(args)

        else:
            print(f"Comando desconhecido: '{line}'. Escreve 'help' para ver os comandos.")


if __name__ == "__main__":
    main()