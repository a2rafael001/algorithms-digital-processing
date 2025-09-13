# lab4_transcode.py
import sys, cv2 as cv
from time import sleep

src = sys.argv[1] if len(sys.argv)>1 else r"C:\Users\rafae\Desktop\4_kurs\kram\lab1\lexuss.mp4"
out = sys.argv[2] if len(sys.argv)>2 else "output.mp4"

cap = cv.VideoCapture(src)
if not cap.isOpened(): raise RuntimeError(f"Не открыть: {src}")

w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 340)
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 180)
fps = cap.get(cv.CAP_PROP_FPS) or 30.0
fourcc = cv.VideoWriter_fourcc(*"mp4v")
writer = cv.VideoWriter(out, fourcc, fps, (w,h))
if not writer.isOpened(): raise RuntimeError(f"Не открыть файл: {out}")

print(f"Транскодирование → {out} ({w}x{h}@{fps:.1f})   Esc=стоп")
while True:
    sleep(0.05)
    ret, frame = cap.read()
    if not ret: break
    writer.write(frame)
    cv.imshow("Preview", frame)
    if (cv.waitKey(1)&0xFF)==27: break

writer.release(); cap.release(); cv.destroyAllWindows()
print("Готово.")
