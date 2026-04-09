# ============================================================
# tracking/logger.py
# MLflow 공통 로거
# ============================================================
import importlib
from typing import List, Optional

import mlflow

from utils.metrics import ExperimentMetrics


# ----------------------------------------------------------
# 내부 헬퍼
# ----------------------------------------------------------

def _get_framework_version(framework: str) -> str:
    """프레임워크 버전 자동 감지"""
    version_map = {
        "sklearn": "sklearn",
        "tensorflow": "tensorflow",
        "flax": "flax",
    }
    try:
        module = importlib.import_module(version_map[framework])
        return getattr(module, "__version__", "unknown")
    except Exception:
        return "unknown"


def _get_device_name(framework: str, cfg: dict) -> str:
    """디바이스 이름 감지"""
    if framework == "flax":
        try:
            import jax
            return jax.default_backend()
        except Exception:
            pass
    return cfg["frameworks"].get(framework, {}).get("jax_backend", "gpu")


# ----------------------------------------------------------
# 공통 로깅
# ----------------------------------------------------------

def log_params(cfg: dict, framework: str) -> None:
    """하이퍼파라미터 로깅 — 세 프레임워크 공통"""
    mlflow.log_params({
        # 프레임워크 / 환경
        "framework": framework,
        "framework_version": _get_framework_version(framework),
        "device": _get_device_name(framework, cfg),
        "random_seed": cfg["data"]["random_seed"],

        # 데이터
        "dataset": cfg["data"]["dataset"],
        "val_split": cfg["data"]["val_split"],

        # 모델 구조
        "model_type": cfg["model"]["model_type"],
        "num_classes": cfg["model"]["num_classes"],
        "filters": str(cfg["model"]["filters"]),
        "dense_units": str(cfg["model"]["dense_units"]),

        # 학습
        "epochs": cfg["train"]["epochs"],
        "batch_size": cfg["train"]["batch_size"],
        "learning_rate": cfg["train"]["learning_rate"],
        "dropout_rate": cfg["train"]["dropout_rate"],
        "optimizer": cfg["train"]["optimizer"],
    })


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
    top5_acc: Optional[float] = None,
    epoch_time: Optional[float] = None,
    throughput: Optional[float] = None,
    grad_norm: Optional[float] = None,
    learning_rate: Optional[float] = None,
) -> None:
    """에포크별 메트릭 로깅"""
    metrics = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
    }
    if val_loss is not None:
        metrics["val_loss"] = val_loss
    if val_acc is not None:
        metrics["val_accuracy"] = val_acc
    if top5_acc is not None:
        metrics["top5_accuracy"] = top5_acc
    if epoch_time is not None:
        metrics["epoch_time_sec"] = epoch_time
    if throughput is not None:
        metrics["throughput_samples_per_sec"] = throughput
    if grad_norm is not None:
        metrics["grad_norm"] = grad_norm
    if learning_rate is not None:
        metrics["learning_rate"] = learning_rate

    mlflow.log_metrics(metrics, step=epoch)


def log_final_metrics(experiment_metrics: ExperimentMetrics) -> None:
    """최종 테스트 지표 및 요약 로깅"""
    m = experiment_metrics
    metrics = {
        "test_accuracy": m.test_acc,
        "top5_accuracy": m.top5_acc,
        "best_val_accuracy": m.best_val_acc,
        "best_epoch": m.best_epoch,
        "avg_epoch_time_sec": m.avg_epoch_time(),
        "avg_throughput_samples_per_sec": m.avg_throughput(),
        "total_train_time_sec": m.total_train_time,
        "first_epoch_time_sec": m.epoch_times[0] if m.epoch_times else 0,
        "peak_memory_mb": m.peak_memory_mb,
        "avg_cpu_pct": m.avg_cpu_pct,
    }
    if m.gpu_memory_used_mb is not None:
        metrics["gpu_memory_used_mb"] = m.gpu_memory_used_mb
    if m.gpu_utilization_pct is not None:
        metrics["gpu_utilization_pct"] = m.gpu_utilization_pct

    mlflow.log_metrics(metrics)

    conv = m.convergence_epoch()
    if conv is not None:
        mlflow.log_metric("convergence_epoch", conv)


def log_artifacts(artifact_paths: List[str]) -> None:
    """아티팩트 파일 로깅"""
    for path in artifact_paths:
        try:
            mlflow.log_artifact(path)
        except Exception as e:
            print(f"[WARN] 아티팩트 로깅 실패 ({path}): {e}")


# ----------------------------------------------------------
# 프레임워크별 전용 로깅
# ----------------------------------------------------------

def log_sklearn_summary(
    n_iter: int,
    converged: bool,
    loss_curve_length: int,
) -> None:
    """Sklearn 전용 수렴 지표 로깅"""
    mlflow.log_metrics({
        "n_iter_actual": n_iter,
        "converged": int(converged),
        "loss_curve_length": loss_curve_length,
    })


def log_tensorflow_summary(
    graph_build_time: float,
    eager_mode: bool,
) -> None:
    """TensorFlow 전용 graph 빌드 지표 로깅"""
    mlflow.log_metrics({
        "graph_build_time_sec": graph_build_time,
    })
    mlflow.log_param("eager_mode", str(eager_mode))


def log_flax_summary(
    epoch_times: List[float],
    xla_compile_time: Optional[float] = None,
) -> None:
    """Flax/JAX 전용 JIT warmup 지표 로깅 — 발표 핵심 지표"""
    if len(epoch_times) < 2:
        return

    epoch1 = epoch_times[0]
    epoch2 = epoch_times[1]
    avg_after_warmup = sum(epoch_times[1:]) / len(epoch_times[1:])
    warmup_overhead = epoch1 - avg_after_warmup
    warmup_ratio = epoch1 / avg_after_warmup if avg_after_warmup > 0 else 0

    metrics = {
        "epoch1_time_sec": epoch1,
        "epoch2_time_sec": epoch2,
        "warmup_overhead_sec": warmup_overhead,
        "warmup_overhead_ratio": warmup_ratio,
        "avg_epoch_time_after_warmup_sec": avg_after_warmup,
    }
    if xla_compile_time is not None:
        metrics["xla_compile_time_sec"] = xla_compile_time

    mlflow.log_metrics(metrics)