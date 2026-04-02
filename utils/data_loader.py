"""공통 데이터 로더 (CIFAR-100 / Iris)"""
import os
from typing import Tuple

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(
    config_path: str = "config.yaml",
    flatten: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    config의 data.dataset에 따라 데이터를 로드합니다.

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test
    """
    cfg = load_config(config_path)
    dataset = cfg["data"]["dataset"]

    if dataset == "iris":
        return _load_iris(cfg, flatten)
    else:
        return _load_cifar100(cfg, flatten)


# --- 하위 호환을 위한 alias ---
def load_cifar100(config_path: str = "config.yaml", flatten: bool = False, normalize: bool = True):
    return load_data(config_path=config_path, flatten=flatten)


def _load_iris(cfg: dict, flatten: bool) -> Tuple:
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    val_split = cfg["data"]["val_split"]
    seed = cfg["data"]["random_seed"]

    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int32)

    # 정규화 (StandardScaler)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # train+val / test 분리 (8:2)
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )
    # train / val 분리
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=val_split, random_state=seed, stratify=y_trainval
    )

    print(f"[INFO] Iris 데이터 로드 완료: "
          f"train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")
    return x_train, y_train, x_val, y_val, x_test, y_test


def _load_cifar100(cfg: dict, flatten: bool) -> Tuple:
    from tensorflow.keras.datasets import cifar100

    val_split = cfg["data"]["val_split"]
    seed = cfg["data"]["random_seed"]
    save_path = os.path.join(cfg["data"]["data_dir"], "cifar100.npz")

    if os.path.exists(save_path):
        data = np.load(save_path)
        x_full = data["x_train"].astype(np.float32) / 255.0
        y_full = data["y_train"].flatten()
        x_test = data["x_test"].astype(np.float32) / 255.0
        y_test = data["y_test"].flatten()
    else:
        (x_full, y_full), (x_test, y_test) = cifar100.load_data()
        x_full = x_full.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        y_full = y_full.flatten()
        y_test = y_test.flatten()

    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=val_split, random_state=seed, stratify=y_full
    )

    if flatten:
        x_train = x_train.reshape(len(x_train), -1)
        x_val   = x_val.reshape(len(x_val), -1)
        x_test  = x_test.reshape(len(x_test), -1)

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_data_info(x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    return {
        "train_size": len(x_train),
        "val_size": len(x_val),
        "test_size": len(x_test),
        "input_shape": x_train.shape[1:],
        "num_classes": len(np.unique(y_train)),
    }
