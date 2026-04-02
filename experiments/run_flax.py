from flax import training
"""Flax/JAX CNN 실험 실행 스크립트"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import mlflow
import optax
import yaml
from flax.training import train_state

from models.flax_model import build_flax_model, cross_entropy_loss, compute_accuracy
from tracking.config import setup_mlflow, get_run_tags
from tracking.logger import log_params, log_epoch_metrics, log_final_metrics
from utils.data_loader import load_cifar100
from utils.metrics import ExperimentMetrics, Timer, get_peak_memory_mb, compute_test_metrics, print_summary


class TrainState(train_state.TrainState):
    batch_stats: any


def create_train_state(model, optimizer, sample_input, rng):
    variables = model.init({"params": rng, "dropout": jax.random.PRNGKey(1)}, sample_input, training=True)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        batch_stats=variables.get("batch_stats", {}),
    )


@jax.jit
def train_step(state, batch_x, batch_y, dropout_rng):
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch_x,
            training=True,
            rngs={"dropout": dropout_rng},
            mutable=["batch_stats"],
        )
        loss = cross_entropy_loss(logits, batch_y)
        return loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    acc = compute_accuracy(logits, batch_y)
    return state, loss, acc


@jax.jit
def eval_step(state, batch_x, batch_y):
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        batch_x,
        training=False,
    )
    loss = cross_entropy_loss(logits, batch_y)
    acc = compute_accuracy(logits, batch_y)
    return loss, acc, logits


def data_generator(x, y, batch_size, rng=None):
    n = len(x)
    indices = np.arange(n)
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield x[idx], y[idx]


def run_flax(config_path: str = "config.yaml") -> ExperimentMetrics:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["data"]["random_seed"]
    rng = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    print("\n[Flax/JAX] 데이터 로딩 중...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar100(config_path=config_path)

    epochs = cfg["train"]["epochs"]
    batch_size = cfg["train"]["batch_size"]

    model, optimizer = build_flax_model(cfg)

    sample = jnp.ones((1, 32, 32, 3))
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(model, optimizer, sample, init_rng)

    setup_mlflow(config_path)
    tags = get_run_tags("flax", config_path)

    with mlflow.start_run(run_name="flax_cnn", tags=tags):
        log_params(cfg, "flax")

        m = ExperimentMetrics(framework="flax")
        shuffle_rng = np.random.default_rng(seed)

        print("[Flax/JAX] 학습 시작 (JIT 컴파일 포함)...")
        mem_before = get_peak_memory_mb()
        total_timer = Timer()
        total_timer.start()

        best_val_acc = 0.0
        no_improve = 0
        patience = 10

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # 학습
            train_losses, train_accs = [], []
            for bx, by in data_generator(x_train, y_train, batch_size, shuffle_rng):
                rng, drop_rng = jax.random.split(rng)
                bx_jnp = jnp.array(bx)
                by_jnp = jnp.array(by)
                state, loss, acc = train_step(state, bx_jnp, by_jnp, drop_rng)
                train_losses.append(float(loss))
                train_accs.append(float(acc))

            # 검증
            val_losses, val_accs = [], []
            for bx, by in data_generator(x_val, y_val, batch_size):
                bx_jnp = jnp.array(bx)
                by_jnp = jnp.array(by)
                loss, acc, _ = eval_step(state, bx_jnp, by_jnp)
                val_losses.append(float(loss))
                val_accs.append(float(acc))

            epoch_time = time.time() - epoch_start
            tr_loss = np.mean(train_losses)
            tr_acc = np.mean(train_accs)
            vl_loss = np.mean(val_losses)
            vl_acc = np.mean(val_accs)

            m.train_losses.append(tr_loss)
            m.train_accs.append(tr_acc)
            m.val_losses.append(vl_loss)
            m.val_accs.append(vl_acc)
            m.epoch_times.append(epoch_time)

            log_epoch_metrics(epoch, tr_loss, tr_acc, vl_loss, vl_acc, epoch_time)
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} | "
                  f"time={epoch_time:.1f}s")

            # Early stopping
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [INFO] Early stopping at epoch {epoch}")
                    break

        m.total_train_time = total_timer.elapsed()
        m.peak_memory_mb = max(get_peak_memory_mb() - mem_before, 0)

        # 테스트 평가
        all_logits = []
        for bx, by in data_generator(x_test, y_test, batch_size):
            bx_jnp = jnp.array(bx)
            by_jnp = jnp.array(by)
            _, _, logits = eval_step(state, bx_jnp, by_jnp)
            all_logits.append(np.array(logits))

        y_proba = np.vstack(all_logits)
        y_pred = np.argmax(y_proba, axis=1)
        test_metrics = compute_test_metrics(y_test, y_pred, y_proba)

        m.test_acc = test_metrics["accuracy"]
        m.top5_acc = test_metrics.get("top5_accuracy", 0.0)

        log_final_metrics(m)
        print_summary(m)

    return m


if __name__ == "__main__":
    run_flax()
