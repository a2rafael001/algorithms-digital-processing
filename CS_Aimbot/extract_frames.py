import cv2
from pathlib import Path

videos_dir = Path("data/raw_videos")
output_dir = Path("data/images/all")
output_dir.mkdir(parents=True, exist_ok=True)

FRAME_STEP = 10
MAP_NAME = "dust2_4"

for video_path in videos_dir.glob("*.*"):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP == 0:
            out_name = f"{MAP_NAME}_{frame_idx:05d}.jpg"
            cv2.imwrite(str(output_dir / out_name), frame)

        frame_idx += 1

    cap.release()

print("Готово!")
