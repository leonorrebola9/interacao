"""
Download de imagens do SKU-110K para classificação manual.
"""
from datasets import load_dataset
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("A descarregar imagens do SKU-110K (streaming)")
dataset = load_dataset("Voxel51/sku110k_test", split="test", streaming=True)

for i, item in enumerate(dataset):
    if i >= 600:
        break

    dest = RAW_DIR / f"img_{i:04d}.jpg"

    if dest.exists():
        print(f"  [{i+1}/600] já existe, a saltar...")
        continue

    img = item["image"].convert("RGB")
    img.save(dest)

    if (i+1) % 10 == 0:
        print(f"  [{i+1}/600] descarregadas...")

print(f"\nConcluído! Imagens em data/raw/")