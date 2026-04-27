import json
import argparse

def generate_report(input_json, output_md):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    report = "# Relatório Semanal de Inteligência de Retalho\n\n"
    report += "## 1. Resumo Executivo\n"
    report += data.get('resumo_executivo', 'Sem resumo disponível.') + "\n\n"
    
    report += "## 2. Insights Detalhados\n"
    for ins in data.get('insights', []):
        report += f"### {ins['titulo']} ({ins['urgencia'].upper()})\n"
        report += f"- **Observação:** {ins['observacao']}\n"
        report += f"- **Implicação:** {ins['implicacao']}\n"
        report += f"- **Recomendação:** {ins['recomendacao']}\n\n"
    
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Relatório final gerado em: {output_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate_report(args.input, args.output)