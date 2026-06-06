from datasets import load_dataset
import os

ds = load_dataset('UniDataPro/grocery-shelves', split='train')

output_dir = './data/images/grocery_shelves'
os.makedirs(output_dir, exist_ok=True)

saved = 0
skipped = 0

for i, item in enumerate(ds):
    img = item['image']
    
    if img.mode == 'RGBA':
        skipped += 1
        continue
    
    # Converte para RGB se necessário (ex: modo P, L, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img.save(f'{output_dir}/grocery_{saved:04d}.jpg')
    saved += 1
    
    if saved % 50 == 0:
        print(f'[{saved}] imagens guardadas...')

print(f'\nFeito! {saved} guardadas, {skipped} ignoradas (RGBA).')
print(f'Total no dataset: {len(ds)}')