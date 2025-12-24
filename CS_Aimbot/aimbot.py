"""
CS 1.6 Aimbot v3.3 - УМНОЕ ПРЕДСКАЗАНИЕ
"""

from ultralytics import YOLO
import mss
import numpy as np
import cv2
import pyautogui
import time
import keyboard

# === НАСТРОЙКИ ===
MODEL_PATH = r"D:\apps\CS_Aimbot\runs\detect\my_train\weights\best_openvino_model"

GAME_WIDTH = 1024
GAME_HEIGHT = 768
CAPTURE_SIZE = 500

CONF_DETECT = 0.35        # ← Изменено
SHOOT_CONF = 0.45         # ← Изменено
IMGSZ = 320

AIM_SPEED = 1.0           # ← Изменено
SHOOT_RADIUS = 120        # ← Изменено
SHOOT_COOLDOWN = 0.05

MIN_TARGET_WIDTH = 20
MIN_TARGET_HEIGHT = 20
MAX_TARGET_WIDTH = 280
MAX_TARGET_HEIGHT = 320
MAX_TARGET_DISTANCE = 220

TORSO_AIM_POINT = 0.40

DIST_PERFECT = 15
DIST_LIKELY = 40
DIST_POSSIBLE = 60

CONSOLE_UPDATE_INTERVAL = 2.0
TOGGLE_KEY = '0'
EXIT_KEY = 'esc'

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

is_active = False
stats = {}
target_history = []

sct = mss.mss()
OFFSET_X = (GAME_WIDTH - CAPTURE_SIZE) // 2
OFFSET_Y = (GAME_HEIGHT - CAPTURE_SIZE) // 2

monitor = {
    "top": OFFSET_Y,
    "left": OFFSET_X,
    "width": CAPTURE_SIZE,
    "height": CAPTURE_SIZE
}

CENTER_X = CAPTURE_SIZE // 2
CENTER_Y = CAPTURE_SIZE // 2

print("🔄 Загрузка модели...")
model = YOLO(MODEL_PATH)

print(f"""
{'='*60}
🎮 CS 1.6 AIMBOT v3.3 - УМНОЕ ПРЕДСКАЗАНИЕ
{'='*60}
⌨️  '{TOGGLE_KEY}' - Вкл/Выкл | '{EXIT_KEY}' - Выход
{'='*60}
""")

last_console_update = 0

def init_stats():
    return {
        'frames': 0,
        'targets_found': 0,
        'shots_fired': 0,
        'start_time': time.time(),
        'last_shot_time': 0,
        'hits': {'perfect': 0, 'likely': 0, 'possible': 0, 'miss': 0}
    }

def predict_position(current_x, current_y):
    global target_history
    
    target_history.append((current_x, current_y))
    
    if len(target_history) > 5:
        target_history.pop(0)
    
    if len(target_history) < 3:
        return current_x, current_y
    
    dx = target_history[-1][0] - target_history[-2][0]
    dy = target_history[-1][1] - target_history[-2][1]
    
    speed = (dx**2 + dy**2)**0.5
    
    if speed < 3:
        return current_x, current_y
    
    predict_factor = min(speed / 20, 1.0)
    
    return current_x + dx * predict_factor, current_y + dy * predict_factor

def select_best_target(detections):
    if not detections:
        return None
    
    best_target = None
    best_score = float('inf')
    
    for x1, y1, x2, y2, conf, cls in detections:
        width = x2 - x1
        height = y2 - y1
        
        if not (MIN_TARGET_WIDTH < width < MAX_TARGET_WIDTH):
            continue
        if not (MIN_TARGET_HEIGHT < height < MAX_TARGET_HEIGHT):
            continue
        
        target_x = (x1 + x2) / 2
        target_y = (y1 + y2) / 2
        
        dist = ((target_x - CENTER_X)**2 + (target_y - CENTER_Y)**2)**0.5
        
        if dist > MAX_TARGET_DISTANCE:
            continue
        
        score = dist / (conf ** 1.5)
        
        if score < best_score:
            best_score = score
            best_target = {'conf': conf, 'box': (x1, y1, x2, y2)}
    
    return best_target

def aim_and_shoot(target):
    global stats
    
    x1, y1, x2, y2 = target['box']
    conf = target['conf']
    
    raw_x = (x1 + x2) / 2
    raw_y = y1 + (y2 - y1) * TORSO_AIM_POINT
    
    target_x, target_y = predict_position(raw_x, raw_y)
    
    dx = target_x - CENTER_X
    dy = target_y - CENTER_Y
    dist = (dx**2 + dy**2)**0.5
    
    speed = AIM_SPEED if dist > 30 else AIM_SPEED * 0.6
    
    move_x = int(dx * speed)
    move_y = int(dy * speed)
    
    if move_x != 0 or move_y != 0:
        pyautogui.move(move_x, move_y, duration=0)
    
    now = time.time()
    if dist < SHOOT_RADIUS and conf >= SHOOT_CONF and now - stats['last_shot_time'] >= SHOOT_COOLDOWN:
        pyautogui.click()
        stats['shots_fired'] += 1
        stats['last_shot_time'] = now
        
        if dist < DIST_PERFECT:
            stats['hits']['perfect'] += 1
            print(f"💥 ✅ | {dist:.0f}px")
        elif dist < DIST_LIKELY:
            stats['hits']['likely'] += 1
            print(f"💥 🎯 | {dist:.0f}px")
        elif dist < DIST_POSSIBLE:
            stats['hits']['possible'] += 1
            print(f"💥 ❓ | {dist:.0f}px")
        else:
            stats['hits']['miss'] += 1
            print(f"💥 ❌ | {dist:.0f}px")

def process_frame():
    global stats, last_console_update, target_history
    
    img = np.array(sct.grab(monitor))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    results = model.predict(img, imgsz=IMGSZ, conf=CONF_DETECT, verbose=False)[0]
    
    stats['frames'] += 1
    
    if results.boxes is not None and len(results.boxes) > 0:
        detections = [
            (b.xyxy[0][0], b.xyxy[0][1], b.xyxy[0][2], b.xyxy[0][3], float(b.conf[0]), int(b.cls[0]))
            for b in results.boxes.cpu().numpy()
        ]
        stats['targets_found'] += len(detections)
        
        target = select_best_target(detections)
        if target:
            aim_and_shoot(target)
        else:
            target_history = []
    else:
        target_history = []
    
    now = time.time()
    if now - last_console_update >= CONSOLE_UPDATE_INTERVAL:
        elapsed = now - stats['start_time']
        fps = stats['frames'] / elapsed if elapsed > 0 else 0
        h = stats['hits']
        total = stats['shots_fired']
        good = h['perfect'] + h['likely']
        acc = (good / total * 100) if total > 0 else 0
        print(f"⚡ FPS: {fps:.0f} | 💥 {total} | 🎯 {acc:.0f}% (✅{h['perfect']} 🎯{h['likely']} ❓{h['possible']} ❌{h['miss']})")
        last_console_update = now

def toggle_aimbot():
    global is_active, stats, target_history
    is_active = not is_active
    
    if is_active:
        stats = init_stats()
        target_history = []
        print(f"\n⚔️  АКТИВИРОВАН\n")
    else:
        print_final_stats()

def print_final_stats():
    h = stats['hits']
    total = stats['shots_fired']
    if total == 0:
        print("\n❌ Нет выстрелов\n")
        return
    
    elapsed = time.time() - stats['start_time']
    fps = stats['frames'] / elapsed
    good = h['perfect'] + h['likely']
    acc = good / total * 100
    
    print(f"""
{'='*60}
📊 СТАТИСТИКА
{'='*60}
⏱️  Время: {elapsed:.0f}с | FPS: {fps:.0f}
💥 Выстрелов: {total} ({total/(elapsed/60):.1f}/мин)

   ✅ Отлично:  {h['perfect']:3d} ({h['perfect']/total*100:.0f}%)
   🎯 Вероятно: {h['likely']:3d} ({h['likely']/total*100:.0f}%)
   ❓ Возможно: {h['possible']:3d} ({h['possible']/total*100:.0f}%)
   ❌ Мимо:     {h['miss']:3d} ({h['miss']/total*100:.0f}%)

🎯 ТОЧНОСТЬ: {acc:.0f}%
{'='*60}
""")

keyboard.add_hotkey(TOGGLE_KEY, toggle_aimbot)
print("⏳ Нажмите '0'")

try:
    while True:
        if keyboard.is_pressed(EXIT_KEY):
            break
        if is_active:
            process_frame()
        else:
            time.sleep(0.1)
except KeyboardInterrupt:
    pass

if is_active:
    print_final_stats()
print("\n✅ Завершено")