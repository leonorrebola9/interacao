import json
import os
from pathlib import Path

IMAGES_DIR = "./data/images"
ANNOTATIONS_DIR = "./data/annotations"
OUTPUT = "./data/dataset.json"

# carrega anotações existentes
annotations = {}
for path in Path(ANNOTATIONS_DIR).glob("*.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    annotations[path.stem] = data  # chave = nome sem extensão

# zonas ciclicas para imagens sem anotação
ZONES = ["Z_S1", "Z_S2", "Z_S3", "Z_S4", "Z_S5", "Z_S6", "Z_S7"]

dataset = {}
zone_counter = 0

for img_path in sorted(Path(IMAGES_DIR).glob("*.jpg")):
    name = img_path.stem  # ex: IMG_8990

    if name in annotations:
        # imagem anotada — usa zona e anotação reais
        ann = annotations[name]
        dataset[img_path.name] = {
            "path": str(img_path),
            "zone": ann.get("zone", "Z_S1"),
            "annotated": True,
            "overall_status": ann.get("overall_status", "ok"),
            "fill_rate": ann.get("fill_rate", 1.0),
            "issues": ann.get("issues", [])
        }
    else:
        # imagem sem anotação — zona ciclica
        zone = ZONES[zone_counter % len(ZONES)]
        zone_counter += 1
        dataset[img_path.name] = {
            "path": str(img_path),
            "zone": zone,
            "annotated": False,
            "overall_status": None,
            "fill_rate": None,
            "issues": []
        }

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

annotated = sum(1 for v in dataset.values() if v["annotated"])
print(f"Feito! {len(dataset)} imagens no dataset.")
print(f"  Anotadas: {annotated}")
print(f"  Sem anotação: {len(dataset) - annotated}")