# -*- coding: utf-8 -*-
import cv2 as cv

def try_open_camera(index=0):
    cap = cv.VideoCapture(index, cv.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv.VideoCapture(index, cv.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv.VideoCapture(index)
    return cap



def main():
    cv.namedWindow("HSV", cv.WINDOW_NORMAL)
    cv.namedWindow("Mask (raw)", cv.WINDOW_NORMAL)
    cv.namedWindow("Mask (morph)", cv.WINDOW_NORMAL)
    cv.namedWindow("Result", cv.WINDOW_NORMAL)


    cap = try_open_camera(0)
    if not cap.isOpened():
        raise RuntimeError("Камера недоступна")

    # пороги по умолчанию 
    H1_low, H1_high = 0, 10
    H2_low, H2_high = 170, 179
    S_low, V_low = 70, 70

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        lower1 = (H1_low, S_low, V_low)
        upper1 = (H1_high, 255, 255)
        lower2 = (H2_low, S_low, V_low)
        upper2 = (H2_high, 255, 255)

        mask1 = cv.inRange(hsv, lower1, upper1)
        mask2 = cv.inRange(hsv, lower2, upper2)
        mask_raw = cv.bitwise_or(mask1, mask2)      # Задание 2

        
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        opened = cv.morphologyEx(mask_raw, cv.MORPH_OPEN, kernel)
        mask_morph = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)

    
        cv.imshow("HSV", hsv)
        cv.imshow("Mask (raw)", mask_raw)
        cv.imshow("Mask (morph)", mask_morph) 

        contours, _ = cv.findContours(mask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        result = frame.copy()
        area = 0
        center = None

        if contours:
            largest = max(contours, key=cv.contourArea)
            area = cv.contourArea(largest)
            M = cv.moments(largest)
            if M["m00"] != 0:
                 cx = int(M["m10"] / M["m00"])
                 cy = int(M["m01"] / M["m00"])
                 center = (cx, cy)
                 cv.drawMarker(result, center, (0, 0, 0), cv.MARKER_CROSS, 12, 2)

        cv.putText(result, f"Area: {int(area)}", (10, 30),
           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv.LINE_AA)
        cv.imshow("Result", result)


        key = cv.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv.destroyAllWindows()
