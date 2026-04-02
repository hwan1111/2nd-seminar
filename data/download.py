"""CIFAR-100 데이터 다운로드 스크립트"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from tensorflow.keras.datasets import cifar100


def download_cifar100(save_dir: str = "data") -> None:
    """CIFAR-100 데이터를 다운로드하고 numpy 파일로 저장합니다."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cifar100.npz")

    if os.path.exists(save_path):
        print(f"[INFO] 이미 다운로드됨: {save_path}")
        return

    print("[INFO] CIFAR-100 다운로드 중...")
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    np.savez_compressed(
        save_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    print(f"[INFO] 저장 완료: {save_path}")
    print(f"  학습 데이터: {x_train.shape}, 레이블: {y_train.shape}")
    print(f"  테스트 데이터: {x_test.shape}, 레이블: {y_test.shape}")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["data"]["data_dir"]
    download_cifar100(save_dir=data_dir)
