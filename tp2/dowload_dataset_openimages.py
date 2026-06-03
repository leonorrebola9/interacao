"""
Fase 2: Extração de anomalias usando o Open Images Dataset.
"""
import os
import io
import json
import time
import hashlib
import glob
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import fiftyone.zoo as foz
import PIL.Image

load_dotenv()
client = genai.Client()

# Pastas e Cache
CATEGORIES = ["empty_shelf", "planogram_violation", "dirty_messy"]
BASE_DIR = Path("data/images")
for cat in CATEGORIES:
    (BASE_DIR / cat).mkdir(parents=True, exist_ok=True)

CACHE_FILE = Path("cache/classification_cache.json")
if CACHE_FILE.exists():
    with open(CACHE_FILE) as f:
        cache = json.load(f)
else:
    cache = {}

def classify_anomaly(image_path: str) -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_hash = hashlib.md5(img_bytes).hexdigest()
    
    if img_hash in cache:
        return cache[img_hash]

    prompt = """Analisa esta imagem e classifica-a numa destas categorias de anomalia de retalho:
    - empty_shelf: prateleira vazia ou com grandes buracos sem produto
    - planogram_violation: produto na posição errada, tombado ou fora do sítio
    - dirty_messy: prateleira muito desorganizada, embalagens danificadas
    - rejeitar: se for uma prateleira normal, se for uma estante de livros de casa, ou se não for um supermercado.
    Responde APENAS com uma destas quatro palavras."""

    try:
        image = PIL.Image.open(io.BytesIO(img_bytes))
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image]
        )
        result = response.text.strip().lower()
        
        if result not in CATEGORIES:
            result = "rejeitar"
            
        cache[img_hash] = result
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
            
        return result
    except Exception as e:
        print(f"  [ERRO API] {e}")
        return "rejeitar"

def run_phase_2():
    print("A descarregar o dataset")
    # Descarrega 1500 imagens brutas para garantir que há problemas suficientes
    raw_dir = "data/open_images_raw"
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=["Shelf"],
        max_samples=1500,
        dataset_dir=raw_dir
    )
    
    targets = {
        "empty_shelf": 100,
        "planogram_violation": 100,
        "dirty_messy": 80
    }
    counts = {cat: 0 for cat in CATEGORIES}
    
    # Procura todas as imagens descarregadas pelo FiftyOne
    image_paths = glob.glob(f"{raw_dir}/train/data/*.jpg")
    print(f"\nForam encontradas {len(image_paths)} imagens brutas prontas para triagem.\n")
    
    for i, path in enumerate(image_paths):
        if all(counts[cat] >= targets[cat] for cat in CATEGORIES):
            print("\nMetas de anomalias atingidas!")
            break
            
        print(f"[{i}/{len(image_paths)}] A procurar anomalias...", end=" ")
        category = classify_anomaly(path)
        print(f"→ {category}")
        
        if category in CATEGORIES and counts[category] < targets[category]:
            img_hash = hashlib.md5(Path(path).read_bytes()).hexdigest()
            dest = BASE_DIR / category / f"oi_{img_hash[:8]}.jpg"
            # Copiar a imagem para a pasta final
            PIL.Image.open(path).save(dest)
            counts[category] += 1
            
        time.sleep(4) # Rate limiting

    print("\n--- Resumo Fase 2 ---")
    for cat, count in counts.items():
        print(f"  {cat}: {count}/{targets[cat]} imagens")

if __name__ == "__main__":
    run_phase_2()