"""TensorFlow CNN 실험 실행 스크립트"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlflow
import mlflow.tensorflow
import yaml

from models.tensorflow_model import build_tensorflow_model
from tracking.config import setup_mlflow, get_run_tags
from tracking.logger import log_params, log_final_metrics
from utils.data_loader import load_data
from utils.metrics import (
    ExperimentMetrics, Timer, CpuMonitor,
    get_peak_memory_mb, compute_throughput, compute_test_metrics, print_summary,
)


class EpochCallback:
    """에폭별 소요 시간, throughput, MLflow 로깅을 담당하는 콜백"""
    import tensorflow as tf

    class _CB(tf.keras.callbacks.Callback):
        def __init__(self, train_size: int):
            super().__init__()
            self.train_size = train_size
            self.epoch_times = []
            self.throughputs = []
            self._start = None

        def on_epoch_begin(self, epoch, logs=None):
            self._start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.time() - self._start
            throughput = self.train_size / elapsed
            self.epoch_times.append(elapsed)
            self.throughputs.append(throughput)

            # 에폭별 MLflow 로깅 (1-indexed step)
            step = epoch + 1
            mlflow.log_metrics({
                "train_loss":               logs.get("loss", 0),
                "train_accuracy":           logs.get("accuracy", 0),
                "val_loss":                 logs.get("val_loss", 0),
                "val_accuracy":             logs.get("val_accuracy", 0),
                "epoch_time_sec":           elapsed,
                "throughput_samples_sec":   throughput,
            }, step=step)


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
        mlflow.tensorflow.autolog(log_every_n_steps=0)
        log_params(cfg, "tensorflow")

        model = build_tensorflow_model(cfg)
        model.summary()

        epochs = cfg["train"]["epochs"]
        batch_size = cfg["train"]["batch_size"]

        time_cb = EpochCallback._CB(train_size=len(x_train))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            time_cb,
        ]

        print("[TensorFlow] 학습 시작...")
        mem_before = get_peak_memory_mb()
        cpu_monitor = CpuMonitor(interval=0.5)
        timer = Timer()
        cpu_monitor.start()
        timer.start()

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        total_time = timer.elapsed()
        avg_cpu = cpu_monitor.stop()
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
        m.test_acc = test_metrics["accuracy"]
        m.top5_acc = test_metrics.get("top5_accuracy", 0.0)

        log_final_metrics(m)
        print_summary(m)

    return m


if __name__ == "__main__":
    run_tensorflow()
