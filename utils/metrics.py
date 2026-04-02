"""공통 평가 지표"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import psutil
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    top_k_accuracy_score,
)


@dataclass
class ExperimentMetrics:
    framework: str
    epoch_times: List[float] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    test_acc: float = 0.0
    top5_acc: float = 0.0
    total_train_time: float = 0.0
    peak_memory_mb: float = 0.0

    def avg_epoch_time(self) -> float:
        return np.mean(self.epoch_times) if self.epoch_times else 0.0

    def convergence_epoch(self, target_acc: float = 0.30) -> Optional[int]:
        """목표 검증 정확도에 처음 도달한 에폭 번호 반환"""
        for i, acc in enumerate(self.val_accs):
            if acc >= target_acc:
                return i + 1
        return None

    def to_dict(self) -> Dict:
        return {
            "framework": self.framework,
            "test_accuracy": self.test_acc,
            "top5_accuracy": self.top5_acc,
            "avg_epoch_time_sec": self.avg_epoch_time(),
            "total_train_time_sec": self.total_train_time,
            "peak_memory_mb": self.peak_memory_mb,
            "convergence_epoch": self.convergence_epoch(),
            "final_val_acc": self.val_accs[-1] if self.val_accs else 0.0,
        }


class Timer:
    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start


def get_peak_memory_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def compute_test_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            metrics["top5_accuracy"] = top_k_accuracy_score(y_true, y_proba, k=5)
        except Exception:
            metrics["top5_accuracy"] = 0.0
    return metrics


def print_summary(metrics: ExperimentMetrics) -> None:
    print(f"\n{'='*50}")
    print(f"  [{metrics.framework}] 실험 결과 요약")
    print(f"{'='*50}")
    print(f"  Test Accuracy   : {metrics.test_acc:.4f} ({metrics.test_acc*100:.2f}%)")
    print(f"  Top-5 Accuracy  : {metrics.top5_acc:.4f} ({metrics.top5_acc*100:.2f}%)")
    print(f"  Avg Epoch Time  : {metrics.avg_epoch_time():.2f}s")
    print(f"  Total Train Time: {metrics.total_train_time:.2f}s")
    print(f"  Peak Memory     : {metrics.peak_memory_mb:.1f} MB")
    conv = metrics.convergence_epoch()
    print(f"  Convergence Ep  : {conv if conv else 'N/A'}")
    print(f"{'='*50}\n")
