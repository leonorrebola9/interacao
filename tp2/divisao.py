import os
import random
import shutil

raw_dirs = [
    "./data/raw/grocery_shelves",
    "./data/raw/sku110k",
    "./data/raw/myphotos"
]

train_dir = "./data/train"
val_dir = "./data/val"
test_dir = "./data/test"

for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

# juntar imagens
all_images = []

for d in raw_dirs:
    for f in os.listdir(d):
        if f.endswith(".jpg"):
            all_images.append(os.path.join(d, f))

random.shuffle(all_images)

# dividir
total = len(all_images)
train_split = int(0.7 * total)
val_split = int(0.85 * total)

train_imgs = all_images[:train_split]
val_imgs = all_images[train_split:val_split]
test_imgs = all_images[val_split:]

def copy_files(files, dst):
    for f in files:
        shutil.copy(f, os.path.join(dst, os.path.basename(f)))

copy_files(train_imgs, train_dir)
copy_files(val_imgs, val_dir)
copy_files(test_imgs, test_dir)

print("Feito! Dataset dividido.")
