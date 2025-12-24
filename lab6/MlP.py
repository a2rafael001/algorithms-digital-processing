import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from CNN import plot_history

# Загрузка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-Hot Encoding
y_train_ohe = to_categorical(y_train, 10)
y_test_ohe = to_categorical(y_test, 10)

# Reshape для augmentation
x_train_aug = x_train.reshape(-1, 28, 28, 1)
x_test_aug = x_test.reshape(-1, 28, 28, 1)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.1
)
datagen.fit(x_train_aug)

# Улучшенная MLP модель
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

start = time.time()
history = model.fit(
    datagen.flow(x_train_aug, y_train_ohe, batch_size=64),
    epochs=30,
    validation_data=(x_test_aug, y_test_ohe),
    callbacks=callbacks,
    verbose=1
)
finish = time.time() - start

print(f"\nВремя обучения: {finish:.2f} сек")

# Оценка
test_loss, test_acc = model.evaluate(x_test_aug, y_test_ohe, verbose=0)
print(f"MLP Test Accuracy: {test_acc:.4f}")

plot_history(history)

model.save('mnist_mlp.keras')
print("Модель MLP сохранена в 'mnist_mlp.keras'")