"""실험 결과 시각화"""
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FRAMEWORK_COLORS = {
    "sklearn": "#4C72B0",
    "tensorflow": "#DD8452",
    "flax": "#55A868",
}


def plot_training_curves(
    metrics_dict: Dict,
    save_dir: str = "results",
    show: bool = False,
) -> None:
    """프레임워크별 학습/검증 정확도 곡선 비교"""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for fw, m in metrics_dict.items():
        color = FRAMEWORK_COLORS.get(fw, None)
        epochs = range(1, len(m.train_accs) + 1)

        if m.train_accs:
            axes[0].plot(epochs, m.train_accs, label=f"{fw} (train)", color=color, linestyle="--")
        if m.val_accs:
            axes[0].plot(epochs, m.val_accs, label=f"{fw} (val)", color=color)

        if m.train_losses:
            axes[1].plot(epochs, m.train_losses, label=f"{fw} (train)", color=color, linestyle="--")
        if m.val_losses:
            axes[1].plot(epochs, m.val_losses, label=f"{fw} (val)", color=color)

    axes[0].set_title("Accuracy Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Loss Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("CIFAR-100: Framework Comparison — Training Curves", fontsize=14)
    plt.tight_layout()

    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    print(f"[INFO] 저장: {path}")
    if show:
        plt.show()
    plt.close()


def plot_comparison_bar(
    metrics_dict: Dict,
    save_dir: str = "results",
    show: bool = False,
) -> None:
    """Test Accuracy / 학습 시간 / 메모리 막대 비교"""
    os.makedirs(save_dir, exist_ok=True)

    frameworks = list(metrics_dict.keys())
    colors = [FRAMEWORK_COLORS.get(fw, "#888888") for fw in frameworks]

    test_accs = [m.test_acc * 100 for m in metrics_dict.values()]
    train_times = [m.total_train_time for m in metrics_dict.values()]
    memories = [m.peak_memory_mb for m in metrics_dict.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def _bar(ax, values, title, ylabel, fmt=".2f"):
        bars = ax.bar(frameworks, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    _bar(axes[0], test_accs, "Test Accuracy (%)", "Accuracy (%)")
    _bar(axes[1], train_times, "Total Training Time (s)", "Seconds", ".1f")
    _bar(axes[2], memories, "Peak Memory Usage (MB)", "MB", ".0f")

    plt.suptitle("CIFAR-100: Framework Comparison Summary", fontsize=14)
    plt.tight_layout()

    path = os.path.join(save_dir, "comparison_bar.png")
    plt.savefig(path, dpi=150)
    print(f"[INFO] 저장: {path}")
    if show:
        plt.show()
    plt.close()


def save_results_csv(metrics_dict: Dict, save_dir: str = "results") -> None:
    """실험 결과를 CSV로 저장"""
    os.makedirs(save_dir, exist_ok=True)
    rows = [m.to_dict() for m in metrics_dict.values()]
    df = pd.DataFrame(rows)
    path = os.path.join(save_dir, "results_summary.csv")
    df.to_csv(path, index=False)
    print(f"[INFO] 결과 CSV 저장: {path}")
    print(df.to_string(index=False))
