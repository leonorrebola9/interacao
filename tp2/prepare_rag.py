import json
import shutil
from pathlib import Path

INSPECTIONS_DIR = "./data/inspections"
RAG_DIR = "./data/inspections_rag"
Path(RAG_DIR).mkdir(exist_ok=True)

copied = 0
skipped = 0

for path in Path(INSPECTIONS_DIR).glob("*.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    
    strategy = data.get("strategy", "")
    image = Path(data.get("image_path", "")).name
    
    # copia SKU (estratégia A) e fotos tuas (estratégia B)
    is_sku = "sku_" in image.lower()
    is_myphotos_b = not is_sku and strategy == "B"
    
    if is_sku or is_myphotos_b:
        shutil.copy(path, Path(RAG_DIR) / path.name)
        copied += 1
    else:
        skipped += 1

print(f"Copiados: {copied}")
print(f"Ignorados: {skipped}")