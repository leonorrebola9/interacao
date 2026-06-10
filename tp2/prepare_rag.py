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
    
    is_sku = "sku_" in image.lower()
    is_myphotos = not is_sku and strategy in ["B", "C"]
    
    if is_sku or is_myphotos:
        shutil.copy(path, Path(RAG_DIR) / path.name)
        copied += 1
    else:
        skipped += 1

print(f"Copiados para RAG: {copied}")
print(f"Ignorados: {skipped}")