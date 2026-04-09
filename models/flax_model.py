"""Flax/JAX CNN 모델 정의 (CIFAR-100)"""
from functools import partial
from typing import Any, Callable

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
        # MaxPool: (2, 2) stride
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not self.training)(x)
        return x


class CIFAR100CNN(nn.Module):
    """TensorFlow 모델과 동일한 CNN 구조 (Flax/JAX)"""
    num_classes: int = 100
    dropout_conv: float = 0.25
    dropout_dense: float = 0.5
    dense_units: int = 512
    training: bool = True

    @nn.compact
    def __call__(self, x):
        x = ConvBlock(32, self.dropout_conv, self.training)(x)
        x = ConvBlock(64, self.dropout_conv, self.training)(x)

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(self.dense_units)(x)
        x = nn.BatchNorm(use_running_average=not self.training)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_dense, deterministic=not self.training)(x)
        x = nn.Dense(self.num_classes)(x)
        return x


def build_flax_model(cfg: dict):
    """Flax 모델 인스턴스 및 optimizer 반환"""
    model_cfg = cfg["model"]
    num_classes = cfg["dataset"]["num_classes"]
    lr = cfg["training"]["learning_rate"]

    model = CIFAR100CNN(
        num_classes=num_classes,
        dropout_conv=model_cfg["dropout_conv"],
        dropout_dense=model_cfg["dropout_dense"],
        dense_units=model_cfg["dense_units"],
        training=True,
    )
    optimizer = optax.adam(learning_rate=lr)
    return model, optimizer


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))


def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> float:
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels).item()
