"""
Fase 1: Extração de prateleiras normais do Grocery Store Dataset.
"""
import os
import json
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from datasets import load_dataset
import PIL.Image
import io

load_dotenv()
client = genai.Client()

# Pastas e Cache
CATEGORIES = ["normal", "ambiguous", "empty_shelf", "planogram_violation", "dirty_messy"]
BASE_DIR = Path("data/images")
for cat in CATEGORIES:
    (BASE_DIR / cat).mkdir(parents=True, exist_ok=True)

CACHE_FILE = Path("cache/classification_cache.json")
CACHE_FILE.parent.mkdir(exist_ok=True)

if CACHE_FILE.exists():
    with open(CACHE_FILE) as f:
        cache = json.load(f)
else:
    cache = {}

def classify_image(img_bytes: bytes, img_hash: str) -> str:
    if img_hash in cache:
        return cache[img_hash]

    prompt = """Analisa esta imagem de uma prateleira de supermercado e classifica-a numa destas categorias:
            - normal: prateleira bem organizada, TODOS os produtos bem posicionados, sem nenhum problema visível
            - empty_shelf: prateleira vazia ou com grandes espaços sem produto
            - planogram_violation: produto tombado, fora do sítio, ou etiqueta ausente
            - dirty_messy: prateleira desorganizada, embalagens danificadas ou produtos desalinhados
            - ambiguous: situação pouco clara ou que mistura vários problemas
            - rejeitar: imagem que não é de prateleira de supermercado
            Responde APENAS com uma destas seis palavras."""

    try:
        image = PIL.Image.open(io.BytesIO(img_bytes))
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image]
        )
        result = response.text.strip().lower()
        
        if result not in ["normal", "ambiguous", "rejeitar"]:
            result = "ambiguous"
            
        cache[img_hash] = result
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
            
        return result
    except Exception as e:
        print(f"  [ERRO API] {e}")
        return "rejeitar"

def run_phase_1():
    print("A carregar o dataset")
    dataset = load_dataset("UniDataPro/grocery-shelves", split="train", streaming=True)
    
    # Metas apenas para estas duas categorias
    targets = {"normal": 150, "ambiguous": 70}
    counts = {"normal": 0, "ambiguous": 0}
    
    processed = 0
    for item in dataset:
        if all(counts[cat] >= targets[cat] for cat in CATEGORIES):
            print("\nMetas do Hugging Face atingidas")
            break
            
        img = item["image"]
        img_byte_arr = io.BytesIO()
        
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        
        print(f"A classificar imagem {processed}", end=" ")
        category = classify_image(img_bytes, img_hash)
        print(f"→ {category}")
        
        if category in CATEGORIES and counts[category] < targets[category]:
            dest = BASE_DIR / category / f"hf_{img_hash[:8]}.jpg"
            img.save(dest)
            counts[category] += 1
            
        processed += 1
        time.sleep(4) # Rate limiting

    print("\n--- Resumo Fase 1 ---")
    for cat, count in counts.items():
        print(f"  {cat}: {count}/{targets[cat]} imagens")

if __name__ == "__main__":
    run_phase_1()