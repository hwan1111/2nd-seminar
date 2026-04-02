"""MLflow 공통 로거"""
from typing import Dict, List, Optional

import mlflow

from utils.metrics import ExperimentMetrics


def log_params(cfg: dict, framework: str) -> None:
    """하이퍼파라미터 로깅"""
    mlflow.log_params({
        "framework": framework,
        "dataset": cfg["data"]["dataset"],
        "num_classes": cfg["model"]["num_classes"],
        "epochs": cfg["train"]["epochs"],
        "batch_size": cfg["train"]["batch_size"],
        "learning_rate": cfg["train"]["learning_rate"],
        "validation_split": cfg["data"]["val_split"],
        "seed": cfg["data"]["random_seed"],
    })


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
    epoch_time: Optional[float] = None,
) -> None:
    """에폭별 메트릭 로깅"""
    metrics = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
    }
    if val_loss is not None:
        metrics["val_loss"] = val_loss
    if val_acc is not None:
        metrics["val_accuracy"] = val_acc
    if epoch_time is not None:
        metrics["epoch_time_sec"] = epoch_time

    mlflow.log_metrics(metrics, step=epoch)


def log_final_metrics(experiment_metrics: ExperimentMetrics) -> None:
    """최종 테스트 지표 및 요약 로깅"""
    mlflow.log_metrics({
        "test_accuracy": experiment_metrics.test_acc,
        "top5_accuracy": experiment_metrics.top5_acc,
        "avg_epoch_time_sec": experiment_metrics.avg_epoch_time(),
        "total_train_time_sec": experiment_metrics.total_train_time,
        "peak_memory_mb": experiment_metrics.peak_memory_mb,
    })

    conv = experiment_metrics.convergence_epoch()
    if conv is not None:
        mlflow.log_metric("convergence_epoch", conv)


def log_artifacts(artifact_paths: List[str]) -> None:
    """아티팩트 파일 로깅"""
    for path in artifact_paths:
        try:
            mlflow.log_artifact(path)
        except Exception as e:
            print(f"[WARN] 아티팩트 로깅 실패 ({path}): {e}")
