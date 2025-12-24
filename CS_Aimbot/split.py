import os
import shutil
import random
from pathlib import Path

# === ТВОИ ПУТИ ===
OLD_TRAIN_IMG = r"D:\apps\CS_Aimbot\data\images\train"
OLD_TRAIN_LBL = r"D:\apps\CS_Aimbot\data\labels\train"
OLD_VAL_IMG = r"D:\apps\CS_Aimbot\data\images\val"
OLD_VAL_LBL = r"D:\apps\CS_Aimbot\data\labels\val"

NEW_IMG = r"D:\apps\CS_Aimbot\auto_collected\images"  # После split
NEW_LBL = r"D:\apps\CS_Aimbot\auto_collected\labels"

# Куда сложим ВСЁ
FINAL_TRAIN_IMG = r"D:\apps\CS_Aimbot\final_dataset\images\train"
FINAL_TRAIN_LBL = r"D:\apps\CS_Aimbot\final_dataset\labels\train"
FINAL_VAL_IMG = r"D:\apps\CS_Aimbot\final_dataset\images\val"
FINAL_VAL_LBL = r"D:\apps\CS_Aimbot\final_dataset\labels\val"

# Создаём папки
for folder in [FINAL_TRAIN_IMG, FINAL_TRAIN_LBL, FINAL_VAL_IMG, FINAL_VAL_LBL]:
    os.makedirs(folder, exist_ok=True)

print("📦 Собираю все файлы...")

# Собираем ВСЕ файлы
all_files = []

# Старые train
for img in Path(OLD_TRAIN_IMG).glob("*.jpg"):
    lbl = Path(OLD_TRAIN_LBL) / (img.stem + ".txt")
    if lbl.exists():
        all_files.append((str(img), str(lbl)))

# Старые val
for img in Path(OLD_VAL_IMG).glob("*.jpg"):
    lbl = Path(OLD_VAL_LBL) / (img.stem + ".txt")
    if lbl.exists():
        all_files.append((str(img), str(lbl)))

# Новые (только если уже в auto_collected после split)
for img in Path(NEW_IMG).glob("*.jpg"):
    lbl = Path(NEW_LBL) / (img.stem + ".txt")
    if lbl.exists():
        all_files.append((str(img), str(lbl)))

print(f"✅ Найдено файлов: {len(all_files)}")

# Перемешиваем
random.shuffle(all_files)

# Делим 80/20
split_idx = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

print(f"\n📊 Разделение:")
print(f"   Train: {len(train_files)}")
print(f"   Val: {len(val_files)}")

# Копируем train
print("\n📁 Копирую train...")
for img_path, lbl_path in train_files:
    img_name = Path(img_path).name
    lbl_name = Path(lbl_path).name
    
    shutil.copy(img_path, os.path.join(FINAL_TRAIN_IMG, img_name))
    shutil.copy(lbl_path, os.path.join(FINAL_TRAIN_LBL, lbl_name))

# Копируем val
print("📁 Копирую val...")
for img_path, lbl_path in val_files:
    img_name = Path(img_path).name
    lbl_name = Path(lbl_path).name
    
    shutil.copy(img_path, os.path.join(FINAL_VAL_IMG, img_name))
    shutil.copy(lbl_path, os.path.join(FINAL_VAL_LBL, lbl_name))

print(f"""
{'='*60}
✅ ГОТОВО!
{'='*60}
📁 Финальный датасет:
   {FINAL_TRAIN_IMG}
   {FINAL_TRAIN_LBL}
   {FINAL_VAL_IMG}
   {FINAL_VAL_LBL}

📊 Статистика:
   Train: {len(train_files)} файлов
   Val: {len(val_files)} файлов
   Всего: {len(all_files)} файлов
{'='*60}
""")
