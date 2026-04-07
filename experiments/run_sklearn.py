# ======================================================
# experiments/run_sklearn.py
# Scikit-learn MLP 실험 실행 스크립트
# ======================================================

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlflow
import yaml

from models.model_registry import get_sklearn_builder
from tracking.config import setup_mlflow, get_run_tags
from tracking.logger import log_params, log_epoch_metrics, log_final_metrics, log_sklearn_summary, log_artifacts
from utils.data_loader import load_data
from utils.metrics import (
    ExperimentMetrics, Timer,
    get_peak_memory_mb, compute_test_metrics, print_summary,
)
from utils.visualize import (
    plot_single_loss_curve, plot_single_accuracy_curve, plot_single_epoch_time,
)


def run_sklearn(config_path: str = "config.yaml") -> ExperimentMetrics:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    artifact_dir = cfg.get("logging", {}).get("artifact_dir", "./artifacts")

    print("\n[Scikit-learn] 데이터 로딩 중...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(
        config_path=config_path, flatten=True
    )

    x_train_full = np.concatenate([x_train, x_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    setup_mlflow(config_path)
    tags = get_run_tags("sklearn", config_path)
    run_name = cfg["mlflow"]["run_names"]["sklearn"]

    with mlflow.start_run(run_name=run_name, tags=tags):
        log_params(cfg, "sklearn")

        build_sklearn_model = get_sklearn_builder(cfg)
        model = build_sklearn_model(cfg)

        print("[Scikit-learn] 학습 시작...")
        timer = Timer()
        timer.start()
        mem_before = get_peak_memory_mb()

        model.fit(x_train_full, y_train_full)

        total_time = timer.elapsed()
        peak_mem = get_peak_memory_mb() - mem_before

        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)
        test_metrics = compute_test_metrics(y_test, y_pred, y_proba)

        m = ExperimentMetrics(framework="sklearn")
        m.total_train_time = total_time
        m.peak_memory_mb = max(peak_mem, 0)
        m.test_acc = test_metrics["accuracy"]
        m.top5_acc = test_metrics.get("top5_accuracy", 0.0)

        if hasattr(model, "loss_curve_"):
            m.train_losses = model.loss_curve_
        if hasattr(model, "validation_scores_"):
            m.val_accs = list(model.validation_scores_)

        n_iters = model.n_iter_
        avg_iter_time = total_time / n_iters if n_iters > 0 else 0.0
        m.epoch_times = [avg_iter_time] * n_iters

        for i, loss in enumerate(m.train_losses):
            val_acc = m.val_accs[i] if i < len(m.val_accs) else None
            log_epoch_metrics(
                epoch=i + 1,
                train_loss=loss,
                train_acc=0.0,
                val_acc=val_acc,
                epoch_time=avg_iter_time,
            )

        m.update_best()

        log_final_metrics(m)
        log_sklearn_summary(
            n_iter=n_iters,
            converged=model.n_iter_ < model.max_iter,
            loss_curve_length=len(m.train_losses),
        )

        # 시각화 및 artifact 저장
        paths = [
            plot_single_loss_curve(m, artifact_dir),
            plot_single_accuracy_curve(m, artifact_dir),
            plot_single_epoch_time(m, artifact_dir),
        ]
        log_artifacts([p for p in paths if p])

        print_summary(m)

    return m


def parse_args():
    parser = argparse.ArgumentParser(description="Sklearn 실험")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sklearn(args.config)