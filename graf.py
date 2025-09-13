# lab5_hsv_two_windows.py
from pathlib import Path
import cv2 as cv

IMG = Path(r"C:\Users\rafae\Desktop\4_kurs\kram\lab1\001.jpg")
bgr = cv.imread(str(IMG), cv.IMREAD_COLOR)
if bgr is None or bgr.size==0: raise FileNotFoundError(IMG)

hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
cv.imshow("BGR (исходное)", bgr)
cv.imshow("HSV (визуализация)", hsv)
print("Любая клавиша…"); cv.waitKey(0); cv.destroyAllWindows()
# lab5_hsv_two_windows.py
from pathlib import Path
import cv2 as cv

IMG = Path(r"C:\path\to\image.jpg")
bgr = cv.imread(str(IMG), cv.IMREAD_COLOR)
if bgr is None or bgr.size==0: raise FileNotFoundError(IMG)

hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
cv.imshow("BGR (исходное)", bgr)
cv.imshow("HSV (визуализация)", hsv)
print("Любая клавиша…"); cv.waitKey(0); cv.destroyAllWindows()
