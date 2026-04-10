"""Flax/JAX ResNet-style 모델 정의 (CIFAR-100)"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class ResBlock(nn.Module):
    """Residual Block: Conv → BN → ReLU → Conv → BN + skip connection"""
    features: int
    dropout_rate: float
    downsample: bool = False  # stride=2로 spatial 축소 여부

    @nn.compact
    def __call__(self, x, training: bool = True):
        stride = (2, 2) if self.downsample else (1, 1)

        # main path
        residual = x
        x = nn.Conv(self.features, (3, 3), strides=stride, padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not training)(x)

        # skip connection: 채널 수 또는 spatial 크기가 다를 때 projection
        if self.downsample or residual.shape[-1] != self.features:
            residual = nn.Conv(self.features, (1, 1), strides=stride, padding="SAME")(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)

        x = nn.relu(x + residual)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        return x


class ResStage(nn.Module):
    """같은 채널 수의 ResBlock을 n_blocks개 쌓는 stage"""
    features: int
    n_blocks: int
    dropout_rate: float
    downsample_first: bool = True

    @nn.compact
    def __call__(self, x, training: bool = True):
        for i in range(self.n_blocks):
            downsample = (i == 0) and self.downsample_first
            x = ResBlock(
                features=self.features,
                dropout_rate=self.dropout_rate,
                downsample=downsample,
            )(x, training=training)
        return x


class CIFAR100ResNet(nn.Module):
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
    num_classes: int = 100
    filters: tuple = (64, 128, 256, 512)
    n_blocks: int = 2
    dense_units: tuple = (1024, 512)
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Stem
        x = nn.Conv(self.filters[0], (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        # Residual stages
        for i, f in enumerate(self.filters):
            x = ResStage(
                features=f,
                n_blocks=self.n_blocks,
                dropout_rate=self.dropout_rate * 0.5,
                downsample_first=(i > 0),   # 첫 stage는 downsample 안 함
            )(x, training=training)

        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))

        # Classifier head
        for units in self.dense_units:
            x = nn.Dense(units)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(self.num_classes)(x)
        return x


def build_flax_model(cfg: dict):
    model = CIFAR100ResNet(
        num_classes=cfg["data"]["num_classes"],
        filters=tuple(cfg["model"]["filters"]),
        n_blocks=cfg["model"].get("n_blocks", 2),
        dense_units=tuple(cfg["model"]["dense_units"]),
        dropout_rate=cfg["train"]["dropout_rate"],
    )
    optimizer = optax.adam(learning_rate=cfg["train"]["learning_rate"])
    return model, optimizer


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))


def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> float:
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)