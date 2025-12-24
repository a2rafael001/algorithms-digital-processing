# -*- coding: utf-8 -*-


import cv2 as cv
import numpy as np
from datetime import datetime


# --- Вспомогательное:  разные бэкенды камеры
def try_open_camera(index=0):
    cap = cv.VideoCapture(index, cv.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv.VideoCapture(index, cv.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv.VideoCapture(index)  # дефолт
    return cap


# --- Трекбар-колбэк 
def _nop(_): pass


def main():
    # Окна
    cv.namedWindow("HSV", cv.WINDOW_NORMAL)
    cv.namedWindow("Mask (raw)", cv.WINDOW_NORMAL)
    cv.namedWindow("Mask (morph)", cv.WINDOW_NORMAL)
    cv.namedWindow("Result", cv.WINDOW_NORMAL)

    # --- ТРЕКБАРЫ ДЛЯ ПОДБОРА ПАРАМЕТРОВ ---
   
    cv.createTrackbar("H1_low",  "Mask (raw)",  0,   179, _nop)   # первая зона (низкие H)
    cv.createTrackbar("H1_high", "Mask (raw)", 10,   179, _nop)

    cv.createTrackbar("H2_low",  "Mask (raw)", 170, 179, _nop)   # вторая зона (высокие H)
    cv.createTrackbar("H2_high", "Mask (raw)", 179, 179, _nop)

    cv.createTrackbar("S_low",   "Mask (raw)", 70,  255, _nop)
    cv.createTrackbar("V_low",   "Mask (raw)", 70,  255, _nop)

    # Морфология: размер ядра (квадратное), минимум 1, нечётное удобнее
    cv.createTrackbar("Kernel",  "Mask (morph)", 5,  31,  _nop)  

    # Порог минимальной площади контура, чтобы отбрасывать шум
    cv.createTrackbar("MinArea","Result", 800, 30000, _nop)

    cap = try_open_camera(0)
    if not cap.isOpened():
        raise RuntimeError("Камера недоступна")

    print("Запущено. Поднесите КРАСНЫЙ объект к камере и подберите пороги на трекбарах.")
    print("Клавиши: Esc/q — выход, s — сохранить кадр.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ---------- Задание 1: HSV ----------
        # OpenCV получает кадр в BGR. Переводим в HSV для более устойчивой цветовой фильтрации.
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # ---------- Задание 2: inRange (красная маска) ----------
        H1_low  = cv.getTrackbarPos("H1_low",  "Mask (raw)")
        H1_high = cv.getTrackbarPos("H1_high", "Mask (raw)")
        H2_low  = cv.getTrackbarPos("H2_low",  "Mask (raw)")
        H2_high = cv.getTrackbarPos("H2_high", "Mask (raw)")
        S_low   = cv.getTrackbarPos("S_low",   "Mask (raw)")
        V_low   = cv.getTrackbarPos("V_low",   "Mask (raw)")

        # Красный цвет охватываем двумя диапазонами по Hue:
        lower1 = np.array([H1_low, S_low, V_low], dtype=np.uint8)
        upper1 = np.array([H1_high, 255, 255],    dtype=np.uint8)
        lower2 = np.array([H2_low, S_low, V_low], dtype=np.uint8)
        upper2 = np.array([H2_high, 255, 255],    dtype=np.uint8)

        mask1 = cv.inRange(hsv, lower1, upper1)
        mask2 = cv.inRange(hsv, lower2, upper2)
        mask_raw = cv.bitwise_or(mask1, mask2)  # объединяем обе «красные» дуги

        # ---------- Задание 3: морфология (открытие + закрытие) 
        k = cv.getTrackbarPos("Kernel", "Mask (morph)")
        k = max(1, k)
        if k % 2 == 0:
            k += 1  # делаем нечётным — визуально часто получается аккуратнее
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (k, k))

        # Открытие: убрать мелкий шум (erode -> dilate)
        opened = cv.morphologyEx(mask_raw, cv.MORPH_OPEN, kernel)
        # Закрытие: залатать маленькие дырки внутри объекта (dilate -> erode)
        mask_morph = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)

        # ---------- Задание 4: моменты и площадь ----------
        # Берём самый большой контур на маске — это и будет наш «красный объект»
        contours, _ = cv.findContours(mask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        result = frame.copy()
        min_area = cv.getTrackbarPos("MinArea", "Result")

        center = None
        bbox = None
        area = 0

        if contours:
            # Находим контур с максимальной площадью
            largest = max(contours, key=cv.contourArea)
            area = cv.contourArea(largest)

            if area >= min_area:
                # Моменты 1-го порядка (у OpenCV: m00, m10, m01...)
                M = cv.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)

                # ---------- Задание 5: прямоугольник вокруг объекта ----------
                x, y, w, h = cv.boundingRect(largest)
                bbox = (x, y, w, h)
                # Чёрный прямоугольник
                cv.rectangle(result, (x, y), (x + w, y + h), (0, 0, 0), 2)
                # Нарисуем центр (для наглядности)
                if center is not None:
                    cv.drawMarker(result, center, (0, 0, 0),
                                  markerType=cv.MARKER_CROSS, markerSize=12, thickness=2)

        # --- Подписи / вывод этапов ---
        cv.putText(result, f"Area: {int(area)}  MinArea: {min_area}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv.LINE_AA)
        if center:
            cv.putText(result, f"Center: {center}", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv.LINE_AA)

        # Показ всех стадий
        cv.imshow("HSV", hsv)
        cv.imshow("Mask (raw)", mask_raw)
        cv.imshow("Mask (morph)", mask_morph)
        cv.imshow("Result", result)

        # Клавиатура
        key = cv.waitKey(1) & 0xFF
        if key in (27, ord('q')):   # Esc / q
            break
        elif key == ord('s'):
            name = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv.imwrite(name, result)
            print(f"Сохранено: {name}")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
