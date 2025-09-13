# lab7_webcam_record.py
import cv2 as cv

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened(): raise RuntimeError("Нет вебкамеры")
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 640)
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 480)
fps = cap.get(cv.CAP_PROP_FPS) or 30.0

writer = cv.VideoWriter("webcam_out.mp4", cv.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
if not writer.isOpened(): raise RuntimeError("Не открыть файл для записи")

print("Запись… Esc=стоп")
while True:
    ret, frame = cap.read()
    if not ret: break
    writer.write(frame)
    cv.imshow("Webcam", frame)
    if (cv.waitKey(1)&0xFF)==27: break

writer.release(); cap.release(); cv.destroyAllWindows()
print("Сохранено: webcam_out.mp4")
