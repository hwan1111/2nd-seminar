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
from utils.data_loader import load_cifar100
from utils.metrics import ExperimentMetrics, Timer, get_peak_memory_mb, compute_test_metrics, print_summary


class EpochTimeCallback:
    """에폭별 소요 시간 기록 콜백"""
    import tensorflow as tf

    class _CB(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_times = []
            self._start = None

        def on_epoch_begin(self, epoch, logs=None):
            self._start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_times.append(time.time() - self._start)


def run_tensorflow(config_path: str = "config.yaml") -> ExperimentMetrics:
    import tensorflow as tf

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["data"]["random_seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    print("\n[TensorFlow] 데이터 로딩 중...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar100(config_path=config_path)

    setup_mlflow(config_path)
    tags = get_run_tags("tensorflow", config_path)

    with mlflow.start_run(run_name="tensorflow_cnn", tags=tags):
        mlflow.tensorflow.autolog(log_every_n_steps=0)
        log_params(cfg, "tensorflow")

        model = build_tensorflow_model(cfg)
        model.summary()

        epochs = cfg["train"]["epochs"]
        batch_size = cfg["train"]["batch_size"]

        time_cb = EpochTimeCallback._CB()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            time_cb,
        ]

        print("[TensorFlow] 학습 시작...")
        mem_before = get_peak_memory_mb()
        timer = Timer()
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
        peak_mem = get_peak_memory_mb() - mem_before

        # 테스트 평가
        y_proba = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
        test_metrics = compute_test_metrics(y_test, y_pred, y_proba)

        m = ExperimentMetrics(framework="tensorflow")
        m.train_losses = history.history.get("loss", [])
        m.train_accs = history.history.get("accuracy", [])
        m.val_losses = history.history.get("val_loss", [])
        m.val_accs = history.history.get("val_accuracy", [])
        m.epoch_times = time_cb.epoch_times
        m.total_train_time = total_time
        m.peak_memory_mb = max(peak_mem, 0)
        m.test_acc = test_metrics["accuracy"]
        m.top5_acc = test_metrics.get("top5_accuracy", 0.0)

        log_final_metrics(m)
        print_summary(m)

    return m


if __name__ == "__main__":
    run_tensorflow()
