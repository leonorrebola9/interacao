'''
colocar no relatório: foi usado o modelo 2.5, pois o modelo 1.5 
já não estava disponível na listagem de modelos da minha API gratuita
'''

import os
from google import genai
from dotenv import load_dotenv

# 1. Carregar as variáveis de ambiente
load_dotenv()
client = genai.Client()

print("Teste")
try:
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents="Olá! Responde apenas 'API configurada com sucesso!' se estiveres a receber esta mensagem."
    )
    print("Sucesso! A resposta do modelo foi:")
    print("---")
    print(response.text)
    print("---")
except Exception as e:
    print(f"Deu erro no teste: {e}")