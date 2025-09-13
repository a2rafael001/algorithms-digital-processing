# lab2_show_images_flags.py
from pathlib import Path
import cv2 as cv

IMG_PATHS = [
    Path(r"C:\Users\rafae\Desktop\4_kurs\kram\lab1\1.jpg"),
    Path(r"C:\Users\rafae\Desktop\4_kurs\kram\lab1\2.png"),
    Path(r"C:\Users\rafae\Desktop\4_kurs\kram\lab1\3.bmp"),
]

READ_FLAGS = {
    "COLOR": cv.IMREAD_COLOR,
    "GRAYSCALE": cv.IMREAD_GRAYSCALE,
    "UNCHANGED": cv.IMREAD_UNCHANGED,
}
WINDOW_FLAGS = {
    "AUTOSIZE": cv.WINDOW_AUTOSIZE,
    "NORMAL": cv.WINDOW_NORMAL,
    "FULLSCREEN": cv.WINDOW_FULLSCREEN,
}

def safe_imshow(win, img, wflag):
    cv.namedWindow(win, wflag)
    if img is None or img.size == 0:
        print(f"[SKIP] {win}: изображение пустое")
        return
    cv.imshow(win, img)
    print(f"[OK] {win}  (любая клавиша)")
    cv.waitKey(0)
    cv.destroyWindow(win)

def main():
    for p in IMG_PATHS:
        if not p.exists():
            print(f"[NOFILE] {p}")
            continue
        for rf_name, rf in READ_FLAGS.items():
            img = cv.imread(str(p), rf)
            if img is None or img.size == 0:
                print(f"[FAIL] {p} + {rf_name}")
                continue
            for wf_name, wf in WINDOW_FLAGS.items():
                safe_imshow(f"{p.name}|READ={rf_name}|WIN={wf_name}", img, wf)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
