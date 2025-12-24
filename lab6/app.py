import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

mlp_model = load_model('mnist_mlp.keras')
cnn_model = load_model('mnist_cnn.keras')


def preprocess_image(img_array):
    """Предобработка как в MNIST: центрирование и нормализация"""
    # Бинаризация
    _, thresh = cv2.threshold(img_array, 30, 255, cv2.THRESH_BINARY)

    # Находим контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros((28, 28), dtype='float32')

    # Находим bounding box
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Вырезаем цифру
    digit = thresh[y:y + h, x:x + w]

    # Масштабируем, сохраняя пропорции (цифра должна быть ~20x20)
    target_size = 20
    if h > w:
        new_h = target_size
        new_w = max(1, int(w * target_size / h))
    else:
        new_w = target_size
        new_h = max(1, int(h * target_size / w))

    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Центрируем в 28x28
    result = np.zeros((28, 28), dtype='uint8')
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    result[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = digit_resized

    # Нормализация
    return result.astype('float32') / 255.0


def predict_digit(img, model_choice):
    if img is None:
        return {str(i): 0.0 for i in range(10)}, None

    # Извлекаем изображение из dict
    if isinstance(img, dict):
        composite = img.get("composite")
        if composite is not None:
            img = composite
        else:
            img = img.get("background")

    if img is None:
        return {str(i): 0.0 for i in range(10)}, None

    # Конвертируем в numpy array
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Конвертируем в grayscale
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA
            # Используем альфа-канал или конвертируем
            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:  # RGB
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Инвертируем (MNIST: белая цифра на чёрном фоне)
    gray = 255 - gray

    # Предобработка
    processed = preprocess_image(gray)

    # Визуализация того, что видит модель
    preview = (processed * 255).astype('uint8')
    preview_img = Image.fromarray(preview).resize((140, 140), Image.Resampling.NEAREST)

    # Предсказание
    if model_choice == "CNN":
        img_input = processed.reshape(1, 28, 28, 1)
        pred = cnn_model.predict(img_input, verbose=0)
    else:
        img_input = processed.reshape(1, 28, 28, 1)
        pred = mlp_model.predict(img_input, verbose=0)

    return {str(i): float(pred[0][i]) for i in range(10)}, preview_img


# Интерфейс
with gr.Blocks(title="Распознавание цифр") as demo:
    gr.Markdown("# Распознавание рукописных цифр — MLP vs CNN")
    gr.Markdown("Нарисуйте цифру от 0 до 9 и выберите модель!")

    with gr.Row():
        with gr.Column(scale=2):
            paint = gr.Paint(label="Нарисуйте цифру")
            model_choice = gr.Radio(
                ["MLP", "CNN"],
                value="MLP",
                label="Выберите модель"
            )
            submit_btn = gr.Button("Распознать", variant="primary")
            clear_btn = gr.Button("Очистить")

        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=5, label="Результат")
            preview = gr.Image(label="Что видит модель (28x28)", height=150)

    submit_btn.click(
        fn=predict_digit,
        inputs=[paint, model_choice],
        outputs=[output_label, preview]
    )

    clear_btn.click(
        fn=lambda: (None, None, None),
        outputs=[paint, output_label, preview]
    )

demo.launch()