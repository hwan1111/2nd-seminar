"""Flax/JAX ResNet-style 모델 정의 (CIFAR-100) - NNX"""
import jax
import jax.numpy as jnp
from flax import nnx
import optax


class ResBlock(nnx.Module):
    """Residual Block: Conv → BN → ReLU → Conv → BN + skip connection"""

    def __init__(self, in_features: int, features: int, dropout_rate: float,
                 downsample: bool = False, rngs: nnx.Rngs = nnx.Rngs(0)):
        stride = (2, 2) if downsample else (1, 1)

        self.conv1 = nnx.Conv(in_features, features, kernel_size=(3, 3),
                              strides=stride, padding="SAME", rngs=rngs)
        self.bn1 = nnx.BatchNorm(features, rngs=rngs)
        self.conv2 = nnx.Conv(features, features, kernel_size=(3, 3),
                              strides=(1, 1), padding="SAME", rngs=rngs)
        self.bn2 = nnx.BatchNorm(features, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        # skip connection projection
        self.need_proj = downsample or (in_features != features)
        if self.need_proj:
            self.proj_conv = nnx.Conv(in_features, features, kernel_size=(1, 1),
                                      strides=stride, padding="SAME", rngs=rngs)
            self.proj_bn = nnx.BatchNorm(features, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x

        x = nnx.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.need_proj:
            residual = self.proj_bn(self.proj_conv(residual))

        x = nnx.relu(x + residual)
        x = self.dropout(x)
        return x


class ResStage(nnx.Module):
    """같은 채널 수의 ResBlock을 n_blocks개 쌓는 stage"""

    def __init__(self, in_features: int, features: int, n_blocks: int,
                 dropout_rate: float, downsample_first: bool = True,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        self.blocks = []
        for i in range(n_blocks):
            downsample = (i == 0) and downsample_first
            block_in = in_features if i == 0 else features
            self.blocks.append(
                ResBlock(
                    in_features=block_in,
                    features=features,
                    dropout_rate=dropout_rate,
                    downsample=downsample,
                    rngs=rngs,
                )
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x)
        return x


class CIFAR100ResNet(nnx.Module):
    """
    ResNet-style model for CIFAR-100 (NNX)

    Architecture:
        Stem: Conv(64) → BN → ReLU
        Stage1: ResBlock x2 (64ch, no downsample)
        Stage2: ResBlock x2 (128ch, downsample)
        Stage3: ResBlock x2 (256ch, downsample)
        Stage4: ResBlock x2 (512ch, downsample)
        GlobalAvgPool → Dense(1024) → Dense(512) → Dense(100)
    """

    def __init__(
        self,
        num_classes: int = 100,
        filters: tuple = (64, 128, 256, 512),
        n_blocks: int = 2,
        dense_units: tuple = (1024, 512),
        dropout_rate: float = 0.5,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        # Stem
        self.stem_conv = nnx.Conv(3, filters[0], kernel_size=(3, 3),
                                  strides=(1, 1), padding="SAME", rngs=rngs)
        self.stem_bn = nnx.BatchNorm(filters[0], rngs=rngs)

        # Residual stages
        self.stages = []
        stage_in = filters[0]
        for i, f in enumerate(filters):
            self.stages.append(
                ResStage(
                    in_features=stage_in,
                    features=f,
                    n_blocks=n_blocks,
                    dropout_rate=dropout_rate * 0.5,
                    downsample_first=(i > 0),
                    rngs=rngs,
                )
            )
            stage_in = f

        # Classifier head
        self.dense_layers = []
        self.bn_layers = []
        self.dropouts = []
        head_in = filters[-1]
        for units in dense_units:
            self.dense_layers.append(nnx.Linear(head_in, units, rngs=rngs))
            self.bn_layers.append(nnx.BatchNorm(units, rngs=rngs))
            self.dropouts.append(nnx.Dropout(rate=dropout_rate, rngs=rngs))
            head_in = units

        self.head = nnx.Linear(head_in, num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Stem
        x = nnx.relu(self.stem_bn(self.stem_conv(x)))

        # Residual stages
        for stage in self.stages:
            x = stage(x)

        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))

        # Classifier head
        for dense, bn, dropout in zip(self.dense_layers, self.bn_layers, self.dropouts):
            x = nnx.relu(bn(dense(x)))
            x = dropout(x)

        return self.head(x)


def build_flax_model(cfg: dict):
    rngs = nnx.Rngs(cfg["data"]["random_seed"])
    model = CIFAR100ResNet(
        num_classes=cfg["model"]["num_classes"],
        filters=tuple(cfg["model"]["filters"]),
        n_blocks=cfg["model"].get("n_blocks", 2),
        dense_units=tuple(cfg["model"]["dense_units"]),
        dropout_rate=cfg["train"]["dropout_rate"],
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=cfg["train"]["learning_rate"]))
    return model, optimizer


# def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
#     one_hot = jax.nn.one_hot(labels, logits.shape[-1])
#     return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarrary:
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))


def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> float:
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)