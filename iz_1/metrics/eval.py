import argparse, os, glob, csv, math
import numpy as np
import pandas as pd

def load_log(path):
    df = pd.read_csv(path)
    # ожидание колонок: frame, ok, x, y, w, h, score, dt_ms
    return df

def metrics_for(df: pd.DataFrame):
    # фильтруем удачные кадры
    ok_mask = df["ok"] == 1
    df_ok = df[ok_mask].copy()
    frames_total = int(df.shape[0])
    frames_ok = int(df_ok.shape[0])

    # fps по среднему времени OK-кадров
    fps = float(1000.0 / df_ok["dt_ms"].mean()) if frames_ok else 0.0

    # time-to-first-failure (в кадрах)
    ttf = frames_total
    if (df["ok"] == 0).any():
        first_fail_idx = int(df.index[(df["ok"] == 0)][0])
        # но index может быть не непрерывный; перейдём по порядку
        fail_row = df.reset_index(drop=True)
        pos = int(fail_row.index[fail_row["ok"] == 0][0])
        ttf = pos

    # число провалов (переходы OK->FAIL)
    ok_series = df["ok"].values.astype(np.int32)
    fails = int(np.sum((ok_series[:-1] == 1) & (ok_series[1:] == 0)))

    # центры и площадь — только на OK-кадрах
    cx = df_ok["x"].values + df_ok["w"].values * 0.5
    cy = df_ok["y"].values + df_ok["h"].values * 0.5
    area = df_ok["w"].values * df_ok["h"].values

    # дрожание центра (стандартное отклонение скорости центра между кадрами)
    if len(cx) >= 3:
        dc = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2)
        jitter_std = float(np.std(dc))
    else:
        jitter_std = float('nan')

    # нестабильность масштаба (STD процента к медиане)
    if len(area) >= 2:
        med = np.median(area)
        scale_std_pct = float(100.0 * np.std(area/med - 1.0))
    else:
        scale_std_pct = float('nan')

    return dict(
        frames_total=frames_total,
        frames_ok=frames_ok,
        fps=round(fps, 2),
        ttf=ttf,
        fails=fails,
        jitter_std_px=None if math.isnan(jitter_std) else round(jitter_std, 2),
        scale_std_pct=None if math.isnan(scale_std_pct) else round(scale_std_pct, 2),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="Папка с CSV логами или glob-паттерн")
    ap.add_argument("--out",  required=True, help="CSV итогов")
    args = ap.parse_args()

    paths = []
    if os.path.isdir(args.logs):
        paths = sorted(glob.glob(os.path.join(args.logs, "*.csv")))
    else:
        paths = sorted(glob.glob(args.logs))

    rows = []
    for p in paths:
        df = load_log(p)
        m = metrics_for(df)
        # извлечём video и method из имени лога: v1_csrt.csv
        base = os.path.splitext(os.path.basename(p))[0]
        # предполагается схема: Name_Method
        parts = base.rsplit("_", 1)
        video = parts[0]
        method = parts[1] if len(parts) > 1 else "unknown"
        rows.append(dict(video=video, method=method, **m))

    out_df = pd.DataFrame(rows)
    out_df.sort_values(["video","method"], inplace=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(out_df)

if __name__ == "__main__":
    main()
