from src.shelf_inspector import inspect_image
from pathlib import Path
import time

ZONES = ["Z_S1", "Z_S2", "Z_S3", "Z_S4", "Z_S5", "Z_S6", "Z_S7"]
images = list(Path("./data/raw/sku110k").glob("*.jpg"))[:50]

for i, img in enumerate(images):
    zone = ZONES[i % len(ZONES)]
    print(f"[{i+1}/50] {img.name} → {zone}")
    try:
        result = inspect_image(str(img), zone_id=zone, strategy="A")
        print(f"  {result.get('overall_status')} | fill_rate: {result.get('shelf_fill_rate')}")
    except Exception as e:
        print(f"  ERRO: {e}")
    time.sleep(5)

print("\nConcluído!")