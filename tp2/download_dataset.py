"""
Classificação manual interactiva do SKU-110K.
"""
import io
import hashlib
from pathlib import Path
from datasets import load_dataset
import PIL.Image
import PIL.ImageTk
import tkinter as tk

CATEGORIES = {
    "1": "normal",
    "2": "empty_shelf",
    "3": "planogram_violation",
    "4": "dirty_messy",
    "5": "ambiguous",
    "0": "rejeitar"
}

BASE_DIR = Path("data/images")
for cat in CATEGORIES.values():
    if cat != "rejeitar":
        (BASE_DIR / cat).mkdir(parents=True, exist_ok=True)

targets = {
    "normal": 150,
    "empty_shelf": 100,
    "planogram_violation": 100,
    "dirty_messy": 80,
    "ambiguous": 70
}

def run():
    counts = {cat: sum(1 for _ in (BASE_DIR / cat).glob("*.jpg"))
              for cat in targets}

    print("A carregar dataset...")
    dataset = load_dataset("Voxel51/sku110k_test", split="test", streaming=True)

    # Criar janela uma vez e manter aberta
    root = tk.Tk()
    root.title("Classificador de Prateleiras")
    root.geometry("900x700")

    img_label = tk.Label(root)
    img_label.pack(pady=5)

    info_label = tk.Label(root, text="", font=("Arial", 11))
    info_label.pack()

    legend = tk.Label(root,
        text="1=normal  2=empty_shelf  3=planogram_violation  4=dirty_messy  5=ambiguous  0=rejeitar  q=sair",
        font=("Arial", 10), fg="gray")
    legend.pack(pady=5)

    result = {"key": None, "waiting": True}

    def on_key(event):
        result["key"] = event.char
        result["waiting"] = False

    root.bind("<Key>", on_key)

    iterator = iter(dataset)
    processed = [0]

    def show_next():
        if all(counts[cat] >= targets[cat] for cat in targets):
            info_label.config(text="Todos os mínimos atingidos! Podes fechar.")
            return

        try:
            item = next(iterator)
        except StopIteration:
            info_label.config(text="Dataset esgotado!")
            return

        img = item["image"].convert("RGB")
        img_display = img.copy()
        img_display.thumbnail((850, 550))

        photo = PIL.ImageTk.PhotoImage(img_display)
        img_label.config(image=photo)
        img_label.image = photo

        status = " | ".join([f"{k}:{counts[k]}/{targets[k]}" for k in targets])
        info_label.config(text=f"Imagem {processed[0]+1} | {status}")

        result["img"] = img
        result["waiting"] = True

        def check_key():
            if not result["waiting"]:
                key = result["key"]

                if key == "q":
                    print("A sair...")
                    root.destroy()
                    return

                if key in CATEGORIES:
                    category = CATEGORIES[key]
                    img_atual = result["img"]

                    if category != "rejeitar" and counts.get(category, 0) < targets.get(category, 0):
                        img_byte_arr = io.BytesIO()
                        img_atual.save(img_byte_arr, format='JPEG')
                        img_hash = hashlib.md5(img_byte_arr.getvalue()).hexdigest()
                        dest = BASE_DIR / category / f"sku_{img_hash[:8]}.jpg"
                        img_atual.save(dest)
                        counts[category] += 1
                        print(f"  [{processed[0]+1}] → {category} ({counts[category]}/{targets.get(category,'?')})")
                    elif category == "rejeitar":
                        print(f"  [{processed[0]+1}] → rejeitado")

                    processed[0] += 1
                    show_next()
                else:
                    # Tecla inválida, aguarda outra
                    result["waiting"] = True
                    root.after(100, check_key)
                    return

            else:
                root.after(100, check_key)

        root.after(100, check_key)

    show_next()
    root.mainloop()

    print("\n--- Resumo final ---")
    for cat, count in counts.items():
        status = "✓" if count >= targets[cat] else f"⚠ faltam {targets[cat]-count}"
        print(f"  {cat}: {count}/{targets[cat]} {status}")

if __name__ == "__main__":
    run()