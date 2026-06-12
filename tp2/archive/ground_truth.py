'''
Ficheiro para juntar todas as anotações feitas
'''

import json
from pathlib import Path

ANNOTATIONS_DIR = "./data/annotations/all_images"
OUTPUT = "./data/annotations/ground_truth.json"

ground_truth = {}

for path in sorted(Path(ANNOTATIONS_DIR).glob("*.json")):
    if path.name == "ground_truth.json":
        continue  # evita incluir o próprio ficheiro de output se já existir

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    image_name = data.get("image", path.stem + ".jpg")
    ground_truth[image_name] = data

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=2, ensure_ascii=False)

print(f"Feito! {len(ground_truth)} imagens no ground truth.")
for img, ann in ground_truth.items():
    status = ann.get("overall_status", "?")
    n_issues = len(ann.get("issues", []))
    print(f"  {img} — {status} ({n_issues} issues)")