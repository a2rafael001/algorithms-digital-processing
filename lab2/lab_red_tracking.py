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
    cap = try_open_camera(0)
    if not cap.isOpened():
        raise RuntimeError("Камера недоступна")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # Задание 1
        cv.imshow("HSV", hsv)
        key = cv.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
