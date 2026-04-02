"""CIFAR-100 공통 데이터 로더"""
import os
from typing import Tuple

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar100


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_cifar100(
    config_path: str = "config.yaml",
    flatten: bool = False,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    CIFAR-100 데이터를 로드하고 전처리합니다.

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test
    """
    cfg = load_config(config_path)
    val_split = cfg["data"]["val_split"]
    seed = cfg["data"]["random_seed"]

    save_path = os.path.join(cfg["data"]["data_dir"], "cifar100.npz")

    if os.path.exists(save_path):
        data = np.load(save_path)
        x_train_full = data["x_train"]
        y_train_full = data["y_train"].flatten()
        x_test = data["x_test"]
        y_test = data["y_test"].flatten()
    else:
        (x_train_full, y_train_full), (x_test, y_test) = cifar100.load_data()
        y_train_full = y_train_full.flatten()
        y_test = y_test.flatten()

    if normalize:
        x_train_full = x_train_full.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=val_split,
        random_state=seed,
        stratify=y_train_full,
    )

    if flatten:
        x_train = x_train.reshape(len(x_train), -1)
        x_val = x_val.reshape(len(x_val), -1)
        x_test = x_test.reshape(len(x_test), -1)

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_data_info(x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    return {
        "train_size": len(x_train),
        "val_size": len(x_val),
        "test_size": len(x_test),
        "input_shape": x_train.shape[1:],
        "num_classes": len(np.unique(y_train)),
    }
