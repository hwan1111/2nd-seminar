"""공통 평가 지표"""
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import psutil
from sklearn.metrics import accuracy_score, top_k_accuracy_score


@dataclass
class ExperimentMetrics:
    framework: str
    epoch_times: List[float] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    throughputs: List[float] = field(default_factory=list)   # samples/sec per epoch
    test_acc: float = 0.0
    top5_acc: float = 0.0
    total_train_time: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_pct: float = 0.0      # 학습 중 평균 CPU 점유율
    jit_warmup_sec: float = 0.0   # Flax: 첫 에폭 시간 (JIT 컴파일 포함)
    steady_epoch_time: float = 0.0  # Flax: 2번째 에폭부터의 평균 시간

    def avg_epoch_time(self) -> float:
        return np.mean(self.epoch_times) if self.epoch_times else 0.0

    def avg_throughput(self) -> float:
        return np.mean(self.throughputs) if self.throughputs else 0.0

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
            "avg_throughput_samples_sec": self.avg_throughput(),
            "total_train_time_sec": self.total_train_time,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_cpu_pct": self.avg_cpu_pct,
            "jit_warmup_sec": self.jit_warmup_sec,
            "steady_epoch_time_sec": self.steady_epoch_time,
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


class CpuMonitor:
    """백그라운드 스레드로 CPU 사용률을 주기적으로 샘플링"""

    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._samples: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._samples.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        """모니터링 중단 후 평균 CPU 점유율(%) 반환"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return float(np.mean(self._samples)) if self._samples else 0.0

    def _sample(self):
        while self._running:
            self._samples.append(psutil.cpu_percent(interval=self._interval))


def get_peak_memory_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def compute_throughput(n_samples: int, elapsed_sec: float) -> float:
    """에폭당 처리 속도 (samples/sec)"""
    return n_samples / elapsed_sec if elapsed_sec > 0 else 0.0


def compute_test_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict:
    metrics = {"accuracy": accuracy_score(y_true, y_pred)}
    if y_proba is not None:
        try:
            metrics["top5_accuracy"] = top_k_accuracy_score(y_true, y_proba, k=5)
        except Exception:
            metrics["top5_accuracy"] = 0.0
    return metrics


def print_summary(metrics: ExperimentMetrics) -> None:
    print(f"\n{'='*55}")
    print(f"  [{metrics.framework}] 실험 결과 요약")
    print(f"{'='*55}")
    print(f"  Test Accuracy      : {metrics.test_acc:.4f} ({metrics.test_acc*100:.2f}%)")
    print(f"  Top-5 Accuracy     : {metrics.top5_acc:.4f} ({metrics.top5_acc*100:.2f}%)")
    print(f"  Avg Epoch Time     : {metrics.avg_epoch_time():.2f}s")
    print(f"  Avg Throughput     : {metrics.avg_throughput():,.0f} samples/sec")
    print(f"  Total Train Time   : {metrics.total_train_time:.2f}s")
    print(f"  Peak Memory        : {metrics.peak_memory_mb:.1f} MB")
    print(f"  Avg CPU Usage      : {metrics.avg_cpu_pct:.1f}%")
    if metrics.jit_warmup_sec > 0:
        print(f"  JIT Warmup (ep 1)  : {metrics.jit_warmup_sec:.2f}s")
        print(f"  Steady Epoch Time  : {metrics.steady_epoch_time:.2f}s")
        speedup = metrics.jit_warmup_sec / metrics.steady_epoch_time if metrics.steady_epoch_time > 0 else 0
        print(f"  JIT Overhead ratio : {speedup:.1f}x (warmup vs steady)")
    conv = metrics.convergence_epoch()
    print(f"  Convergence Ep     : {conv if conv else 'N/A'}")
    print(f"{'='*55}\n")
