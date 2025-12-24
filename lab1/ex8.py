# lab8_cross_fill_by_center_rgb.py
import cv2 as cv
import numpy as np

def nearest_rgb_primary(bgr):
    # OpenCV: BGR → RGB
    r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
    primaries = {"RED": (255,0,0), "GREEN": (0,255,0), "BLUE": (0,0,255)}
    # евклидово расстояние
    best = min(primaries.items(),
               key=lambda kv: (r-kv[1][0])**2 + (g-kv[1][1])**2 + (b-kv[1][2])**2)
    name, rgb = best
    # вернуть BGR для отрисовки:
    return name, (rgb[2], rgb[1], rgb[0])

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened(): raise RuntimeError("Нет вебкамеры")

thick = -1      # -1 = заливка
arm = 180
bar = 30        # толщина перекладины

alpha = 0.5     # полупрозрачная заливка

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]; cx, cy = w//2, h//2

    # определить цвет по центральному пикселю
    name, color = nearest_rgb_primary(frame[cy, cx])
    overlay = frame.copy()

    # залитый крест (две прямоугольные перекладины)
    cv.rectangle(overlay, (cx-arm, cy-bar//2), (cx+arm, cy+bar//2), color, thick)
    cv.rectangle(overlay, (cx-bar//2, cy-arm), (cx+bar//2, cy+arm), color, thick)

    # смешиваем с исходником для мягкости
    frame = cv.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    # контур красный поверх (как на примере)
    cv.rectangle(frame, (cx-arm, cy-bar//2), (cx+arm, cy+bar//2), (0,0,255), 2)
    cv.rectangle(frame, (cx-bar//2, cy-arm), (cx+bar//2, cy+arm), (0,0,255), 2)

    cv.putText(frame, f"Center->{name}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
    cv.imshow("Cross filled by nearest RGB", frame)
    if (cv.waitKey(1)&0xFF)==27: break

cap.release(); cv.destroyAllWindows()
