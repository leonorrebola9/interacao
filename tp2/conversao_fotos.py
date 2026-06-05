from PIL import Image
import pillow_heif
import os
from pathlib import Path

pillow_heif.register_heif_opener()

INPUT_DIR = "./data/images/myphotos"
OUTPUT_DIR = "./data/images/myphotos"

converted = 0
erros = 0

heic_files = list(Path(INPUT_DIR).glob("*.heic")) + list(Path(INPUT_DIR).glob("*.HEIC"))
print(f"Encontradas {len(heic_files)} imagens HEIC")

for heic_path in heic_files:
    try:
        jpg_path = Path(OUTPUT_DIR) / (heic_path.stem + ".jpg")
        
        img = Image.open(heic_path)
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img.save(jpg_path, "JPEG", quality=90)
        os.remove(heic_path)  # apaga o HEIC original
        
        converted += 1
        print(f"  {heic_path.name} → {jpg_path.name}")
    
    except Exception as e:
        erros += 1
        print(f"  ERRO em {heic_path.name}: {e}")

print(f"\nFeito! {converted} convertidas, {erros} erros.")