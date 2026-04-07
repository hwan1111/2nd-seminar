# ======================================================
# experiments/run_tensorflow.py
# TensorFlow CNN 실험 실행 스크립트
# ======================================================

import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlflow
import yaml

from models.tensorflow_model import build_tensorflow_model
from tracking.config import setup_mlflow, get_run_tags
from tracking.logger import log_params, log_final_metrics, log_tensorflow_summary
from utils.data_loader import load_data
from utils.metrics import (
    ExperimentMetrics, Timer, CpuMonitor, GpuMonitor,
    get_peak_memory_mb, compute_throughput, compute_test_metrics, print_summary,
)


def _make_epoch_callback(train_size: int, cfg: dict):
    import tensorflow as tf

    class EpochCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_times = []
            self.throughputs = []
            self._start = None

        def on_epoch_begin(self, epoch, logs=None):
            self._start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.time() - self._start
            throughput = compute_throughput(train_size, elapsed)
            self.epoch_times.append(elapsed)
            self.throughputs.append(throughput)

            # lr scheduler 사용 시 현재 lr 추출
            lr = None
            try:
                lr = float(self.model.optimizer.learning_rate)
            except Exception:
                pass

            metrics = {
                "train_loss":             logs.get("loss", 0),
                "train_accuracy":         logs.get("accuracy", 0),
                "val_loss":               logs.get("val_loss", 0),
                "val_accuracy":           logs.get("val_accuracy", 0),
                "epoch_time_sec":         elapsed,
                "throughput_samples_sec": throughput,
            }
            if lr is not None:
                metrics["learning_rate"] = lr

            mlflow.log_metrics(metrics, step=epoch + 1)

    return EpochCallback()


def run_tensorflow(config_path: str = "config.yaml") -> ExperimentMetrics:
    import tensorflow as tf

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["data"]["random_seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    print("\n[TensorFlow] 데이터 로딩 중...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(config_path=config_path)

    setup_mlflow(config_path)
    tags = get_run_tags("tensorflow", config_path)

    with mlflow.start_run(run_name="tensorflow_cnn", tags=tags):
        log_params(cfg, "tensorflow")

        model = build_tensorflow_model(cfg)
        model.summary()

        epochs = cfg["train"]["epochs"]
        batch_size = cfg["train"]["batch_size"]
        track_gpu = cfg.get("logging", {}).get("track_gpu", False)

        time_cb = _make_epoch_callback(train_size=len(x_train), cfg=cfg)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            time_cb,
        ]

        print("[TensorFlow] 학습 시작...")
        mem_before = get_peak_memory_mb()
        cpu_monitor = CpuMonitor(interval=0.5)
        gpu_monitor = GpuMonitor() if track_gpu else None
        timer = Timer()

        cpu_monitor.start()
        if gpu_monitor:
            gpu_monitor.start()
        timer.start()

        # graph 빌드 시간 측정 (첫 번째 batch 실행 전후)
        graph_build_start = time.time()
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        graph_build_time = time_cb.epoch_times[0] if time_cb.epoch_times else 0.0

        total_time = timer.elapsed()
        avg_cpu = cpu_monitor.stop()
        gpu_stats = gpu_monitor.stop() if gpu_monitor else {"gpu_memory_used_mb": None, "gpu_utilization_pct": None}
        peak_mem = get_peak_memory_mb() - mem_before

        y_proba = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
        test_metrics = compute_test_metrics(y_test, y_pred, y_proba)

        m = ExperimentMetrics(framework="tensorflow")
        m.train_losses = history.history.get("loss", [])
        m.train_accs = history.history.get("accuracy", [])
        m.val_losses = history.history.get("val_loss", [])
        m.val_accs = history.history.get("val_accuracy", [])
        m.epoch_times = time_cb.epoch_times
        m.throughputs = time_cb.throughputs
        m.total_train_time = total_time
        m.peak_memory_mb = max(peak_mem, 0)
        m.avg_cpu_pct = avg_cpu
        m.gpu_memory_used_mb = gpu_stats["gpu_memory_used_mb"]
        m.gpu_utilization_pct = gpu_stats["gpu_utilization_pct"]
        m.test_acc = test_metrics["accuracy"]
        m.top5_acc = test_metrics.get("top5_accuracy", 0.0)
        m.update_best()

        log_final_metrics(m)
        log_tensorflow_summary(
            graph_build_time=graph_build_time,
            eager_mode=not tf.executing_eagerly(),
        )
        print_summary(m)

    return m


def parse_args():
    parser = argparse.ArgumentParser(description="TensorFlow CNN 실험")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tensorflow(args.config)