"""Flax/JAX CNN 모델 정의 (CIFAR-100)"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class ConvBlock(nn.Module):
    features: int
    dropout_rate: float
    training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not self.training)(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not self.training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not self.training)(x)
        return x


class CIFAR100CNN(nn.Module):
    """TensorFlow 모델과 동일한 CNN 구조 (Flax/JAX)"""
    num_classes: int = 100
    filters: tuple = (32, 64, 128)
    dense_units: tuple = (512, 256)
    dropout_rate: float = 0.5
    training: bool = True

    @nn.compact
    def __call__(self, x):
        for f in self.filters:
            x = ConvBlock(f, self.dropout_rate * 0.5, self.training)(x)

        x = x.reshape((x.shape[0], -1))  # Flatten
        for units in self.dense_units:
            x = nn.Dense(units)(x)
            x = nn.BatchNorm(use_running_average=not self.training)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not self.training)(x)

        x = nn.Dense(self.num_classes)(x)
        return x


def build_flax_model(cfg: dict):
    """Flax 모델 인스턴스 및 optimizer 반환"""
    model = CIFAR100CNN(
        num_classes=cfg["model"]["num_classes"],
        filters=tuple(cfg["model"]["filters"]),
        dense_units=tuple(cfg["model"]["dense_units"]),
        dropout_rate=cfg["train"]["dropout_rate"],
        training=True,
    )
    optimizer = optax.adam(learning_rate=cfg["train"]["learning_rate"])
    return model, optimizer


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))


def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> float:
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels).item()
