"""Scikit-learn MLP 실험 실행 스크립트"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlflow
import mlflow.sklearn
import yaml

from models.sklearn_model import build_sklearn_model
from tracking.config import setup_mlflow, get_run_tags
from tracking.logger import log_params, log_final_metrics
from utils.data_loader import load_cifar100
from utils.metrics import ExperimentMetrics, Timer, get_peak_memory_mb, compute_test_metrics, print_summary


def run_sklearn(config_path: str = "config.yaml") -> ExperimentMetrics:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("\n[Scikit-learn] 데이터 로딩 중...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar100(
        config_path=config_path, flatten=True
    )

    # 학습+검증 합치기 (sklearn은 내부에서 validation_fraction 사용)
    x_train_full = np.concatenate([x_train, x_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    setup_mlflow(config_path)
    tags = get_run_tags("sklearn", config_path)

    with mlflow.start_run(run_name="sklearn_mlp", tags=tags):
        mlflow.sklearn.autolog()
        log_params(cfg, "sklearn")

        model = build_sklearn_model(cfg)

        print("[Scikit-learn] 학습 시작...")
        timer = Timer()
        timer.start()
        mem_before = get_peak_memory_mb()

        model.fit(x_train_full, y_train_full)

        total_time = timer.elapsed()
        peak_mem = get_peak_memory_mb() - mem_before

        # 테스트 평가
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)
        test_metrics = compute_test_metrics(y_test, y_pred, y_proba)

        # ExperimentMetrics 구성
        m = ExperimentMetrics(framework="sklearn")
        m.total_train_time = total_time
        m.peak_memory_mb = max(peak_mem, 0)
        m.test_acc = test_metrics["accuracy"]
        m.top5_acc = test_metrics.get("top5_accuracy", 0.0)

        # sklearn의 loss_curve_ 활용
        if hasattr(model, "loss_curve_"):
            m.train_losses = model.loss_curve_
        if hasattr(model, "validation_scores_"):
            m.val_accs = list(model.validation_scores_)
        # epoch별 시간 근사
        n_iters = model.n_iter_
        m.epoch_times = [total_time / n_iters] * n_iters

        log_final_metrics(m)
        print_summary(m)

    return m


if __name__ == "__main__":
    run_sklearn()
