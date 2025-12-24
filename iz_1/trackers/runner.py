import argparse, time, os, csv, json
import cv2
import numpy as np

# --- Compat layer for different OpenCV builds ---
def _create_tracker(name: str):
    name = name.lower()
    err = []
    # Try modern API
    try:
        if name == "csrt":  return cv2.TrackerCSRT_create()
        if name == "kcf":   return cv2.TrackerKCF_create()
        if name == "mosse": return cv2.TrackerMOSSE_create()
    except Exception as e:
        err.append(f"modern API: {e}")
    # Try legacy API
    try:
        legacy = getattr(cv2, "legacy", None)
        if legacy is not None:
            if name == "csrt":  return legacy.TrackerCSRT_create()
            if name == "kcf":   return legacy.TrackerKCF_create()
            if name == "mosse": return legacy.TrackerMOSSE_create()
    except Exception as e:
        err.append(f"legacy API: {e}")
    raise RuntimeError(f"Cannot create tracker {name}. Tried both APIs. Details: {err}")

def _draw_box(frame, bbox, color=(0,255,0), label:str=""):
    # мягкая проверка формата
    if bbox is None or not hasattr(bbox, "__len__") or len(bbox) != 4:
        return
    try:
        x, y, w, h = [int(round(float(v))) for v in bbox]
    except Exception:
        return

    H, W = frame.shape[:2]
    # минимальные размеры и в кадр
    w = max(2, w)
    h = max(2, h)
    x = max(0, min(x, W - w))
    y = max(0, min(y, H - h))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(frame, label, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()

    # view options
    ap.add_argument("--show", action="store_true", help="показывать окно трекинга (Q/Esc — стоп)")
    ap.add_argument("--view-maxw", type=int, default=1280, help="макс. ширина окна просмотра")
    ap.add_argument("--view-maxh", type=int, default=720,  help="макс. высота окна просмотра")

    ap.add_argument("--method", required=True, choices=["csrt","kcf","mosse","ncc"])
    ap.add_argument("--video", required=True)
    ap.add_argument("--save",  required=True, help="output video path")
    ap.add_argument("--log",   default=None, help="csv log path (default: logs/<name>.csv)")

    # NCC params
    ap.add_argument("--threshold", type=float, default=0.6, help="loss threshold for NCC (0..1)")
    ap.add_argument("--scale",     type=float, default=1.5, help="search window scale (relative to bbox)")

    # ROI cache
    ap.add_argument("--roi-cache", default="logs/roi_cache.json",
                    help="файл с сохранёнными ROI по имени видео")
    ap.add_argument("--use-cached-roi",  action="store_true",
                    help="использовать ROI из кэша (если есть) и не показывать selectROI")
    ap.add_argument("--save-cached-roi", action="store_true",
                    help="после выбора ROI сохранить его в кэш")

    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    writer = cv2.VideoWriter(args.save, fourcc, fps, (W,H))

    # --- first frame for ROI ---
    ok, frame0 = cap.read()
    if not ok:
        raise SystemExit("Empty video.")

    # --- try cache ROI ---
    key = os.path.basename(args.video)
    roi_cache = {}
    if os.path.isfile(args.roi_cache):
        try:
            with open(args.roi_cache, "r", encoding="utf-8") as f:
                roi_cache = json.load(f)
        except Exception:
            roi_cache = {}

    init_bbox = None
    if args.use_cached_roi and key in roi_cache:
        x, y, w, h = map(int, roi_cache[key])
        init_bbox = (x, y, w, h)

    # --- Select ROI (downscaled) if not from cache ---
    if init_bbox is None:
        maxW, maxH = 1280, 720
        scale = min(maxW / W, maxH / H, 1.0)
        disp = frame0 if scale >= 1.0 else cv2.resize(
            frame0, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA
        )
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", disp.shape[1], disp.shape[0])
        print("Select a ROI and then press SPACE or ENTER button!")
        print("Cancel the selection process by pressing c button!")
        init_bbox_disp = cv2.selectROI("Select ROI", disp, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        if init_bbox_disp == (0, 0, 0, 0):
            raise SystemExit("No ROI selected.")
        sx, sy, sw, sh = map(int, init_bbox_disp)
        init_bbox = (
            int(round(sx / scale)),
            int(round(sy / scale)),
            int(round(sw / scale)),
            int(round(sh / scale)),
        )

        # sanitize ROI
        x, y, w0, h0 = init_bbox
        w = max(w0, 48); h = max(h0, 48)
        x = max(0, min(x, W - w)); y = max(0, min(y, H - h))
        init_bbox = (x, y, w, h)

        if args.save_cached_roi:
            roi_cache[key] = [int(v) for v in init_bbox]
            os.makedirs(os.path.dirname(args.roi_cache), exist_ok=True)
            with open(args.roi_cache, "w", encoding="utf-8") as f:
                json.dump(roi_cache, f, ensure_ascii=False, indent=2)

    # --- build tracker + robust init ---
    use_gray = False
    if args.method == "ncc":
        from trackers.ncc_tracker import NCCTracker
        tracker = NCCTracker(threshold=args.threshold, search_scale=args.scale)
        tracker.init(frame0, init_bbox)
    else:
        tracker = _create_tracker(args.method)
        try:
            tracker.init(frame0, init_bbox)
        except cv2.error as e:
            print("[WARN] tracker.init failed, retry with legacy/gray:", e)
            # try legacy builder
            if hasattr(cv2, "legacy"):
                if args.method == "csrt":  tracker = cv2.legacy.TrackerCSRT_create()
                elif args.method == "kcf": tracker = cv2.legacy.TrackerKCF_create()
                elif args.method == "mosse": tracker = cv2.legacy.TrackerMOSSE_create()
            # try gray init
            gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            tracker.init(gray0, init_bbox)
            use_gray = True

    # --- open log ---
    base = os.path.splitext(os.path.basename(args.save))[0]
    log_path = args.log or os.path.join("logs", f"{base}.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["frame","ok","x","y","w","h","score","dt_ms"])

    # start from frame #1 (frame0 уже использован для init)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break


            t0 = time.perf_counter()

            frame_for_tracker = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if use_gray else frame
            if args.method == "ncc":
                ok_tr, bbox, score = tracker.update(frame)  # наш NCC работает по цветному кадру
                # санитизация bbox на всякий случай
                x, y, w, h = [int(round(float(v))) for v in bbox]
                w = max(2, w);
                h = max(2, h)
                x = max(0, min(x, W - w));
                y = max(0, min(y, H - h))
                bbox = (x, y, w, h)
            else:
                ok_tr, bbox = tracker.update(frame_for_tracker)
                score = np.nan

            dt_ms = (time.perf_counter() - t0) * 1000.0


            color = (0,255,0) if ok_tr else (0,0,255)
            label = f"{args.method.upper()}  #{frame_idx}  {'OK' if ok_tr else 'LOST'}"
            if args.method == "ncc":
                label += f"  s={score:.2f}"
            _draw_box(frame, bbox if ok_tr else init_bbox, color, label)
            cv2.putText(frame, f"{dt_ms:.1f} ms", (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            # show (downscaled) preview if requested
            if args.show:
                vscale = min(args.view_maxw / W, args.view_maxh / H, 1.0)
                disp = frame if vscale >= 1.0 else cv2.resize(
                    frame, (int(W * vscale), int(H * vscale)), interpolation=cv2.INTER_AREA
                )
                cv2.imshow("Tracking", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # Esc или q
                    print("[INFO] Остановлено пользователем.")
                    break

            writer.write(frame)

            if frame_idx % max(1, int(fps)) == 0:
                print(f"[{frame_idx}/{total}] {dt_ms:.1f} ms кадр")

            x,y,w,h = bbox if ok_tr else init_bbox
            log.writerow([
                frame_idx, int(ok_tr),
                f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}",
                f"{score if not (isinstance(score, float) and np.isnan(score)) else ''}",
                f"{dt_ms:.3f}"
            ])

            frame_idx += 1
    finally:
        log_f.close()
        writer.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
