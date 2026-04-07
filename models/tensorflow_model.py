"""TensorFlow/Keras ResNet-style 모델 정의 (CIFAR-100)"""
import tensorflow as tf
from tensorflow.keras import layers, models


def _res_block(x, features: int, dropout_rate: float, downsample: bool = False):
    """Residual Block: Conv → BN → ReLU → Conv → BN + skip connection"""
    stride = (2, 2) if downsample else (1, 1)

    # main path
    residual = x
    x = layers.Conv2D(features, (3, 3), strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(features, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)

    # skip connection: 채널 수 또는 spatial 크기가 다를 때 projection
    if downsample or residual.shape[-1] != features:
        residual = layers.Conv2D(features, (1, 1), strides=stride, padding="same")(residual)
        residual = layers.BatchNormalization()(residual)

    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def _res_stage(x, features: int, n_blocks: int, dropout_rate: float, downsample_first: bool = True):
    """같은 채널 수의 ResBlock을 n_blocks개 쌓는 stage"""
    for i in range(n_blocks):
        downsample = (i == 0) and downsample_first
        x = _res_block(x, features, dropout_rate, downsample=downsample)
    return x


def build_tensorflow_model(cfg: dict) -> tf.keras.Model:
    """
    ResNet-style model for CIFAR-100

    Architecture:
        Stem: Conv(64) → BN → ReLU
        Stage1: ResBlock x2 (64ch, no downsample)
        Stage2: ResBlock x2 (128ch, downsample)
        Stage3: ResBlock x2 (256ch, downsample)
        Stage4: ResBlock x2 (512ch, downsample)
        GlobalAvgPool → Dense(1024) → Dense(512) → Dense(100)
    """
    num_classes = cfg["model"]["num_classes"]
    lr = cfg["train"]["learning_rate"]
    dropout_rate = cfg["train"]["dropout_rate"]
    filters = cfg["model"]["filters"]          # [64, 128, 256, 512]
    n_blocks = cfg["model"].get("n_blocks", 2)
    dense_units = cfg["model"]["dense_units"]  # [1024, 512]

    inputs = layers.Input(shape=tuple(cfg["model"]["input_shape"]))
    x = inputs

    # Stem
    x = layers.Conv2D(filters[0], (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual stages
    for i, f in enumerate(filters):
        x = _res_stage(
            x,
            features=f,
            n_blocks=n_blocks,
            dropout_rate=dropout_rate * 0.5,
            downsample_first=(i > 0),   # 첫 stage는 downsample 안 함
        )

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Classifier head
    for units in dense_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="cifar100_resnet_tf")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model