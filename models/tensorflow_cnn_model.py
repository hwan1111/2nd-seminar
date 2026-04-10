"""TensorFlow/Keras CNN 모델 정의 (CIFAR-100)"""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_tensorflow_model(cfg: dict) -> tf.keras.Model:
    """
    Conv2D + MaxPool + Dense 구조의 CNN 모델.

    Architecture:
        Conv(32) → Conv(32) → MaxPool → Dropout
        Conv(64) → Conv(64) → MaxPool → Dropout
        Conv(128) → Conv(128) → MaxPool → Dropout
        Flatten → Dense(512) → Dense(256) → Dense(100, softmax)
    """
    num_classes = cfg["data"]["num_classes"]
    lr = cfg["train"]["learning_rate"]
    dropout_rate = cfg["train"]["dropout_rate"]
    filters = cfg["model"]["filters"]        # [32, 64, 128]
    dense_units = cfg["model"]["dense_units"]  # [512, 256]

    inputs = layers.Input(shape=tuple(cfg["model"]["input_shape"]))
    x = inputs

    for f in filters:
        x = layers.Conv2D(f, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(f, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)  # conv dropout = dropout_rate/2

    x = layers.Flatten()(x)
    for units in dense_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="cifar100_cnn_tf")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
