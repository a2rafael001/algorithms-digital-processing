# lab9_phone_cam.py
import sys, cv2 as cv
url = sys.argv[1] if len(sys.argv)>1 else "http://192.168.1.23:4747/video"  # пример для IP Webcam
cap = cv.VideoCapture(url)
if not cap.isOpened(): raise RuntimeError(f"Не открыть: {url}")
print("Esc=выход")
while True:
    ret, frame = cap.read()
    if not ret: break
    cv.imshow("Phone camera", frame)
    if (cv.waitKey(1)&0xFF)==27: break
cap.release(); cv.destroyAllWindows()
