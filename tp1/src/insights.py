import json
import ollama
import argparse
import os

class InsightEngine:
    def __init__(self, model="llama3.2"):
        self.model = model

    def get_prompt(self, data):
        return f"""
        Analisa estes dados de retalho (JSON): {json.dumps(data)}
        Gera insights estratégicos. 
        Segue obrigatoriamente estes exemplos:
        - "A zona Z_S3 teve 847 visitantes, 31% acima da média."
        - "No domingo às 16h, Z_N4 teve 0 visitantes - possível obstrução."

        Responde APENAS em JSON estruturado com as chaves: 
        "insights" (lista de objetos com titulo, observacao, implicacao, recomendacao, urgencia) 
        e "resumo_executivo" (string).
        """

    def run(self, input_json, output_json):
        if not os.path.exists(input_json):
            print(f"Erro: {input_json} não encontrado!")
            return

        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"A comunicar com Ollama ({self.model})")
        try:
            # A biblioteca oficial trata de toda a comunicação e evita erros 404
            response = ollama.generate(
                model=self.model,
                prompt=self.get_prompt(data),
                format='json',
                options={'temperature': 0}
            )
            
            with open(output_json, 'w', encoding='utf-8') as f:
                f.write(response['response'])
            print(f"Sucesso! Insights guardados em {output_json}")
            
        except Exception as e:
            print(f"Erro: Certifica-te que fizeste 'ollama pull {self.model}' no terminal.")
            print(f"Detalhe: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    InsightEngine().run(args.input, args.output)