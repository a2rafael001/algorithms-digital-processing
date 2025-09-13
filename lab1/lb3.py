# lab3_video_playback.py
import sys, cv2 as cv

def open_source():
    if len(sys.argv) > 1:
        src = sys.argv[1]
        cap = cv.VideoCapture(src)
        name = src
    else:
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # вебка
        name = "Webcam(0)"
    if not cap.isOpened(): raise RuntimeError(f"Не открыть: {name}")
    return cap, name

def main():
    cap, name = open_source()
    mode, scale = "BGR", 1.0
    print("g=GRAY, h=HSV, c=BGR, 1=0.5x, 2=1.0x, Esc=выход")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size==0: break
        if mode=="GRAY": shown = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        elif mode=="HSV": shown = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        else: shown = frame
        if scale!=1.0:
            shown = cv.resize(shown, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        cv.imshow(f"Video: {name}", shown)
        k = cv.waitKey(1) & 0xFF
        if k==27: break
        elif k in (ord('g'),ord('G')): mode="GRAY"
        elif k in (ord('h'),ord('H')): mode="HSV"
        elif k in (ord('c'),ord('C')): mode="BGR"
        elif k==ord('1'): scale=0.5
        elif k==ord('2'): scale=1.0
    cap.release(); cv.destroyAllWindows()

if __name__ == "__main__": main()
