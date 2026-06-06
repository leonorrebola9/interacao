from datasets import load_dataset
from PIL import Image
from io import BytesIO
import os

output_dir = './data/images/sku110k'
os.makedirs(output_dir, exist_ok=True)

TOTAL = 500

print("A carregar dataset em modo streaming (sem descarregar tudo)")
dataset = load_dataset("PrashantDixit0/SKU-110K", split="train", streaming=True)

saved = 0
erros = 0

for i, sample in enumerate(dataset):
    if saved >= TOTAL:
        break
    
    try:
        filename = os.path.join(output_dir, f"sku_{saved:04d}.jpg")
        
        if os.path.exists(filename):
            saved += 1
            continue
        
        img = Image.open(BytesIO(sample["image"]["bytes"]))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.save(filename)
        saved += 1
        
        if saved % 25 == 0:
            print(f"[{saved}/{TOTAL}] guardadas")
    
    except Exception as e:
        erros += 1
        print(f"Erro na imagem {i}: {e}")

print(f"\nFeito! {saved} guardadas, {erros} erros.")