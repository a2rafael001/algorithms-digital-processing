from ultralytics import YOLO


model = YOLO("yolo11n.pt")

model.train(
    data="yolo_cs.yaml",
    epochs=100,
    patience=25,
    imgsz=640,
    batch=4,
    lr0=0.01,
    device='cpu',
    project="runs/detect",
    name="my_train"
)

# Экспорт
best_model = YOLO("runs/detect/my_train/weights/best.pt")
best_model.export(format="openvino", imgsz=320, half=False)

print("✅ Готово!")