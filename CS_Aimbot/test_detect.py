from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = r"D:\apps\CS_Aimbot\runs\detect\retrain\weights\best.pt"
TEST_IMAGES = r"D:\apps\CS_Aimbot\data\images\test"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=TEST_IMAGES,
    save=True,
    conf=0.25,
    save_txt=True,
    project="runs/detect",
    name="test_results"
)

print("\n" + "="*60)
print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
print("="*60)

total_images = 0
total_detections = 0
images_with_detections = 0
confidence_sum = 0

for result in results:
    total_images += 1
    boxes = result.boxes
    
    if boxes is not None and len(boxes) > 0:
        images_with_detections += 1
        num_detections = len(boxes)
        total_detections += num_detections
        
        # Имя файла
        img_name = Path(result.path).name
        
        print(f"\n📷 {img_name}")
        print(f"   Найдено объектов: {num_detections}")
        
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            confidence_sum += conf
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            
            print(f"   [{i+1}] Conf: {conf:.1%} | Размер: {w:.0f}×{h:.0f}px")

print("\n" + "="*60)
print("📈 ИТОГО")
print("="*60)
print(f"Всего изображений: {total_images}")
print(f"С детекциями: {images_with_detections} ({images_with_detections/total_images*100:.1f}%)")
print(f"Без детекций: {total_images - images_with_detections}")
print(f"Всего детекций: {total_detections}")

if total_detections > 0:
    avg_conf = confidence_sum / total_detections
    avg_per_image = total_detections / images_with_detections if images_with_detections > 0 else 0
    print(f"Средний confidence: {avg_conf:.1%}")
    print(f"Среднее детекций на фото: {avg_per_image:.1f}")

print(f"\n📁 Картинки с рамками: runs/detect/test_results/")
print("="*60)
