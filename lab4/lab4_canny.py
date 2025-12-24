import cv2
import numpy as np
from pathlib import Path


def read_and_gaussian_blur(image_path: str,
                           ksize: int = 5,
                           sigma: float = 1.0):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть файл: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)
    return img, gray, blurred


def sobel_gradients(gray_blur: np.ndarray):

    gray_f = gray_blur.astype(np.float64)

    Gx_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    Gy_kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float64)

    h, w = gray_f.shape
    padded = np.pad(gray_f, ((1, 1), (1, 1)), mode='edge')

    Gx = np.zeros_like(gray_f)
    Gy = np.zeros_like(gray_f)

    # Свёртка 3×3
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            region = padded[i - 1:i + 2, j - 1:j + 2]
            Gx[i - 1, j - 1] = np.sum(region * Gx_kernel)
            Gy[i - 1, j - 1] = np.sum(region * Gy_kernel)

    mag = np.hypot(Gx, Gy)      # длина градиента
    angle = np.arctan2(Gy, Gx)  # угол в радианах
    return Gx, Gy, mag, angle


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:

    arr = arr.copy().astype(np.float64)
    min_val = arr.min()
    arr -= min_val
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val * 255.0
    return arr.astype(np.uint8)


def non_max_suppression(mag: np.ndarray, angle: np.ndarray) -> np.ndarray:

    h, w = mag.shape
    res = np.zeros((h, w), dtype=np.float64)

    angle_deg = np.rad2deg(angle)
    angle_deg[angle_deg < 0] += 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = 0.0
            r = 0.0

            # 0°
            if (0 <= angle_deg[i, j] < 22.5) or (157.5 <= angle_deg[i, j] < 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            # 45°
            elif 22.5 <= angle_deg[i, j] < 67.5:
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            # 90°
            elif 67.5 <= angle_deg[i, j] < 112.5:
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            # 135°
            elif 112.5 <= angle_deg[i, j] < 157.5:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]

            if (mag[i, j] >= q) and (mag[i, j] >= r):
                res[i, j] = mag[i, j]
            else:
                res[i, j] = 0.0

    return res


def double_threshold(nms: np.ndarray,
                     mag: np.ndarray,
                     low_ratio: float = 1.0 / 25.0,
                     high_ratio: float = 1.0 / 10.0) -> np.ndarray:
    """
    Двойная пороговая фильтрация.
    low_ratio и high_ratio задают пороги как доли от max(grad).
    """
    max_grad = mag.max()
    high = max_grad * high_ratio
    low = max_grad * low_ratio

    h, w = nms.shape
    result = np.zeros((h, w), dtype=np.uint8)

    strong = nms >= high
    weak = (nms >= low) & (nms < high)

    # Сильные пиксели сразу считаем границами
    result[strong] = 255

    # Простое "наращивание" слабых пикселей рядом с сильными
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if weak[i, j]:
                if np.any(strong[i - 1:i + 2, j - 1:j + 2]):
                    result[i, j] = 255

    return result


# ===============================
# ЗАДАНИЯ 1–4
# ===============================

def task1(image_path: str,
          ksize: int = 5,
          sigma: float = 1.0):
    """
    Задание 1: чтение, ЧБ, Гаусс, вывод.
    """
    orig, gray, blurred = read_and_gaussian_blur(image_path, ksize=ksize, sigma=sigma)

    cv2.imshow("Original (BGR)", orig)
    cv2.imshow("Gray", gray)
    cv2.imshow(f"Gray + Gaussian blur (ksize={ksize}, sigma={sigma})", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task2(image_path: str,
          ksize: int = 5,
          sigma: float = 1.0):
    """
    Задание 2: две матрицы — длина и угол градиента.
    """
    _, _, blurred = read_and_gaussian_blur(image_path, ksize=ksize, sigma=sigma)
    Gx, Gy, mag, angle = sobel_gradients(blurred)

    print("=== Задание 2 ===")
    print("Форма матрицы длины градиента:", mag.shape)
    print("Пример матрицы длины градиента [0:5, 0:5]:")
    print(np.round(mag[0:5, 0:5], 2))

    print("\nФорма матрицы углов градиента:", angle.shape)
    print("Пример матрицы углов (рад) [0:5, 0:5]:")
    print(np.round(angle[0:5, 0:5], 2))

    mag_disp = normalize_to_uint8(mag)
    angle_disp = (angle + np.pi) / (2 * np.pi)  # [0,1]
    angle_disp = (angle_disp * 255).astype(np.uint8)

    cv2.imshow("Gradient magnitude (normalized)", mag_disp)
    cv2.imshow("Gradient angle (pseudo-image)", angle_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task3(image_path: str,
          ksize: int = 5,
          sigma: float = 1.0):
    """
    Задание 3: подавление немаксимумов.
    """
    _, _, blurred = read_and_gaussian_blur(image_path, ksize=ksize, sigma=sigma)
    Gx, Gy, mag, angle = sobel_gradients(blurred)
    nms = non_max_suppression(mag, angle)
    nms_disp = normalize_to_uint8(nms)

    print("=== Задание 3 ===")
    print("Форма матрицы после NMS:", nms.shape)

    cv2.imshow("After Non-Maximum Suppression", nms_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task4(image_path: str,
          ksize: int = 5,
          sigma: float = 1.0,
          low_ratio: float = 1.0 / 25.0,
          high_ratio: float = 1.0 / 10.0):
    """
    Задание 4: двойная пороговая фильтрация.
    """
    orig, gray, blurred = read_and_gaussian_blur(image_path, ksize=ksize, sigma=sigma)
    Gx, Gy, mag, angle = sobel_gradients(blurred)
    nms = non_max_suppression(mag, angle)
    edges = double_threshold(nms, mag, low_ratio=low_ratio, high_ratio=high_ratio)

    print("=== Задание 4 ===")
    print("Максимальное значение градиента:", mag.max())
    print(f"Пороги: low = max*{low_ratio:.4f}, high = max*{high_ratio:.4f}")

    cv2.imshow("Original", orig)
    cv2.imshow("Gray + Gaussian blur", blurred)
    cv2.imshow("Edges after double threshold (manual Canny)", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===============================
# ЗАДАНИЕ 5 — ЭКСПЕРИМЕНТЫ
# ===============================

def task5(image_path: str):

    ksizes = [3, 5, 7]
    sigmas = [0.8, 1.2, 2.0]
    threshold_pairs = [
        (1.0 / 30.0, 1.0 / 15.0),
        (1.0 / 25.0, 1.0 / 10.0),
        (1.0 / 20.0, 1.0 / 8.0),
    ]

    img_path = Path(image_path)
    out_dir = img_path.parent / "canny_experiments"
    out_dir.mkdir(exist_ok=True)

    print("=== Задание 5: эксперименты с параметрами ===")
    print("Результаты будут сохранены в папку:", out_dir)

    for ksize in ksizes:
        for sigma in sigmas:
            _, _, blurred = read_and_gaussian_blur(image_path, ksize=ksize, sigma=sigma)
            Gx, Gy, mag, angle = sobel_gradients(blurred)
            nms = non_max_suppression(mag, angle)

            for low_ratio, high_ratio in threshold_pairs:
                edges = double_threshold(nms, mag,
                                         low_ratio=low_ratio,
                                         high_ratio=high_ratio)

                fname = (
                    f"edges_k{ksize}_s{sigma:.1f}_"
                    f"low{low_ratio:.4f}_high{high_ratio:.4f}.png"
                )
                save_path = out_dir / fname
                cv2.imwrite(str(save_path), edges)

                print(
                    f"Сохранён файл: {fname} "
                    f"(ksize={ksize}, sigma={sigma}, "
                    f"low={low_ratio:.4f}, high={high_ratio:.4f})"
                )

    print("\nОткрой сохранённые PNG и выбери на глаз, где границы выглядят лучше всего.")
    print("Параметры этого файла и будут твоими \"наилучшими\" для отчёта.")


# ===============================
# ТОЧКА ВХОДА
# ===============================

if __name__ == "__main__":
    # !!! УКАЖИ СВОЙ ПУТЬ К ИЗОБРАЖЕНИЮ !!!
    img_path = r"C:\Users\rafae\Desktop\4_kurs\kram\lab4\777.jpg"

    # По очереди запускай то, что нужно:
    #task1(img_path, ksize=5, sigma=1.0)
    #task2(img_path, ksize=5, sigma=1.0)
    #task3(img_path, ksize=5, sigma=1.0)
    #task4(img_path, ksize=5, sigma=1.0,
     #      low_ratio=1.0/25.0, high_ratio=1.0/10.0)
    #task5(img_path)
