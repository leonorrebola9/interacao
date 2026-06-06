import os
import json
from google import genai
from dotenv import load_dotenv
from PIL import Image

# carregar API key
load_dotenv()
client = genai.Client()

images_dir = "./data/train"   
output_dir = "./data/annotations"

os.makedirs(output_dir, exist_ok=True)

PROMPT = """
Analisa esta imagem de uma prateleira de supermercado.

Identifica problemas como:
- empty_shelf
- misaligned
- damaged
- wrong_product

Responde APENAS com JSON neste formato:

{
  "issues": [
    {
      "type": "",
      "severity": "",
      "location": ""
    }
  ],
  "overall_status": "",
  "fill_rate": 0.0
}
"""

for img_name in os.listdir(images_dir):

    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(images_dir, img_name)
    annotation_path = os.path.join(output_dir, img_name.replace(".jpg", ".json"))

    if os.path.exists(annotation_path):
        continue

    print(f"A analisar {img_name}")

    image = Image.open(img_path)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[PROMPT, image]
        )

        import re

        text = response.text.strip()

        # tenta extrair JSON mesmo com texto extra
        match = re.search(r'\{.*\}', text, re.DOTALL)

        if match:
            json_text = match.group(0)
            data = json.loads(json_text)
        else:
            raise ValueError("JSON não encontrado na resposta")
        
        data["image"] = img_name

        with open(annotation_path, "w") as f:
            json.dump(data, f, indent=2)

        print("DEBUG resposta:")
        print(text)

        print(f"Guardado: {img_name}")

    except Exception as e:
        print(f"Erro em {img_name}: {e}")