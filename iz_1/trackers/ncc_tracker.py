# trackers/ncc_tracker.py
import cv2
import numpy as np

class NCCTracker:
    def __init__(self, threshold=0.6, search_scale=1.5):
        self.threshold = float(threshold)
        self.search_scale = float(search_scale)
        self.tpl = None         # шаблон (uint8, 1 канал)
        self.bbox = None        # (x,y,w,h) в координатах исходного кадра

    def init(self, frame, bbox):
        x, y, w, h = map(int, bbox)
        H, W = frame.shape[:2]
        # Санитарные границы
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        roi = frame[y:y+h, x:x+w]
        # Градации серого, uint8 — чтобы matchTemplate не ругался
        self.tpl = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.bbox = (x, y, w, h)

    def _extract_search(self, frame):
        """Вырезаем окно поиска вокруг текущего bbox и конвертируем в gray."""
        x, y, w, h = self.bbox
        cx = x + w // 2
        cy = y + h // 2
        sw = int(round(w * self.search_scale))
        sh = int(round(h * self.search_scale))
        H, W = frame.shape[:2]
        x0 = max(0, cx - sw // 2)
        y0 = max(0, cy - sh // 2)
        x1 = min(W, x0 + sw)
        y1 = min(H, y0 + sh)
        patch = frame[y0:y1, x0:x1]
        search_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return (x0, y0), search_gray

    def _match_multi_scale(self, search_im, scales=(0.95, 1.0, 1.05)):
        """Перебор нескольких масштабов шаблона, оба изображения — uint8, 1ch."""
        best_score = -1.0
        best_pos = (0, 0)
        best_scale = 1.0
        for s in scales:
            tpl = self.tpl if s == 1.0 else cv2.resize(
                self.tpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA
            )
            # если шаблон больше окна поиска — пропускаем
            if tpl.shape[0] > search_im.shape[0] or tpl.shape[1] > search_im.shape[1]:
                continue
            res = cv2.matchTemplate(search_im, tpl, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            if maxv > best_score:
                best_score = float(maxv)
                best_pos = maxloc
                best_scale = s
        return best_score, best_pos, best_scale

    def update(self, frame):
        (sx, sy), search_im = self._extract_search(frame)
        score, (mx, my), s = self._match_multi_scale(
            search_im, scales=(0.9, 0.95, 1.0, 1.05, 1.1)
        )
        ok = score >= self.threshold

        # Обновляем bbox (даже при плохом score оставим старый, чтобы не прыгал)
        x, y, w, h = self.bbox
        if ok:
            w = int(round(w * s))
            h = int(round(h * s))
            x = sx + mx
            y = sy + my
            self.bbox = (x, y, w, h)

        return ok, self.bbox, score
