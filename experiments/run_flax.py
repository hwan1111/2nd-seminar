# ======================================================
# experiments/run_flax.py
# Flax/JAX CNN 실험 실행 스크립트
# ======================================================

import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import mlflow
import yaml
from flax.training import train_state

from models.model_registry import get_flax_builder
from tracking.config import setup_mlflow, get_run_tags
from tracking.logger import log_params, log_epoch_metrics, log_final_metrics, log_flax_summary, log_artifacts
from utils.data_loader import load_data
from utils.metrics import (
    ExperimentMetrics, Timer, CpuMonitor, GpuMonitor,
    get_peak_memory_mb, compute_throughput, compute_test_metrics, print_summary,
)
from utils.visualize import (
    plot_single_loss_curve, plot_single_accuracy_curve,
    plot_single_epoch_time, plot_jit_warmup,
)


class TrainState(train_state.TrainState):
    batch_stats: any


def create_train_state(model, optimizer, sample_input, rng):
    variables = model.init(
        {"params": rng, "dropout": jax.random.PRNGKey(1)},
        sample_input,
        training=True,
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        batch_stats=variables.get("batch_stats", {}),
    )


def make_train_step(cross_entropy_loss, compute_accuracy):
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
        grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        return state, loss, acc, grad_norm
    return train_step


def make_eval_step(cross_entropy_loss, compute_accuracy):
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
    return eval_step


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

    track_gpu = cfg.get("logging", {}).get("track_gpu", False)
    artifact_dir = cfg.get("logging", {}).get("artifact_dir", "./artifacts")

    print("\n[Flax/JAX] 데이터 로딩 중...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(config_path=config_path)

    epochs = cfg["train"]["epochs"]
    batch_size = cfg["train"]["batch_size"]

    build_flax_model, cross_entropy_loss, compute_accuracy = get_flax_builder(cfg)
    train_step = make_train_step(cross_entropy_loss, compute_accuracy)
    eval_step = make_eval_step(cross_entropy_loss, compute_accuracy)

    model, optimizer = build_flax_model(cfg)

    sample = jnp.ones((1, *cfg["model"]["input_shape"]))
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(model, optimizer, sample, init_rng)

    setup_mlflow(config_path)
    tags = get_run_tags("flax", config_path)
    run_name = cfg["mlflow"]["run_names"]["flax"]

    with mlflow.start_run(run_name=run_name, tags=tags):
        log_params(cfg, "flax")

        m = ExperimentMetrics(framework="flax")
        shuffle_rng = np.random.default_rng(seed)

        print("[Flax/JAX] 학습 시작 (JIT 컴파일 포함)...")
        mem_before = get_peak_memory_mb()
        cpu_monitor = CpuMonitor(interval=0.5)
        gpu_monitor = GpuMonitor() if track_gpu else None
        total_timer = Timer()

        cpu_monitor.start()
        if gpu_monitor:
            gpu_monitor.start()
        total_timer.start()

        best_val_acc = 0.0
        no_improve = 0
        patience = 10

        xla_compile_start = time.perf_counter()
        xla_compiled = False

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_losses, train_accs, grad_norms = [], [], []
            for bx, by in data_generator(x_train, y_train, batch_size, shuffle_rng):
                rng, drop_rng = jax.random.split(rng)
                bx_jnp = jnp.array(bx)
                by_jnp = jnp.array(by)
                state, loss, acc, grad_norm = train_step(state, bx_jnp, by_jnp, drop_rng)

                if not xla_compiled:
                    jax.effects_barrier()
                    xla_compile_time = time.perf_counter() - xla_compile_start
                    xla_compiled = True

                train_losses.append(float(loss))
                train_accs.append(float(acc))
                grad_norms.append(float(grad_norm))

            val_losses, val_accs_ep = [], []
            for bx, by in data_generator(x_val, y_val, batch_size):
                bx_jnp = jnp.array(bx)
                by_jnp = jnp.array(by)
                loss, acc, _ = eval_step(state, bx_jnp, by_jnp)
                val_losses.append(float(loss))
                val_accs_ep.append(float(acc))

            epoch_time = time.time() - epoch_start
            throughput = compute_throughput(len(x_train), epoch_time)
            tr_loss = float(np.mean(train_losses))
            tr_acc = float(np.mean(train_accs))
            vl_loss = float(np.mean(val_losses))
            vl_acc = float(np.mean(val_accs_ep))
            avg_grad_norm = float(np.mean(grad_norms))

            m.train_losses.append(tr_loss)
            m.train_accs.append(tr_acc)
            m.val_losses.append(vl_loss)
            m.val_accs.append(vl_acc)
            m.epoch_times.append(epoch_time)
            m.throughputs.append(throughput)

            jit_label = " ← JIT warmup" if epoch == 1 else ""

            log_epoch_metrics(
                epoch=epoch,
                train_loss=tr_loss,
                train_acc=tr_acc,
                val_loss=vl_loss,
                val_acc=vl_acc,
                epoch_time=epoch_time,
                throughput=throughput,
                grad_norm=avg_grad_norm,
            )
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} | "
                  f"time={epoch_time:.1f}s tput={throughput:,.0f} smp/s{jit_label}")

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [INFO] Early stopping at epoch {epoch}")
                    break

        m.total_train_time = total_timer.elapsed()
        m.avg_cpu_pct = cpu_monitor.stop()
        gpu_stats = gpu_monitor.stop() if gpu_monitor else {"gpu_memory_used_mb": None, "gpu_utilization_pct": None}
        m.peak_memory_mb = max(get_peak_memory_mb() - mem_before, 0)
        m.gpu_memory_used_mb = gpu_stats["gpu_memory_used_mb"]
        m.gpu_utilization_pct = gpu_stats["gpu_utilization_pct"]

        m.update_best()
        m.update_jit_stats()

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
        log_flax_summary(
            epoch_times=m.epoch_times,
            xla_compile_time=xla_compile_time if xla_compiled else None,
        )

        # 시각화 및 artifact 저장
        paths = [
            plot_single_loss_curve(m, artifact_dir),
            plot_single_accuracy_curve(m, artifact_dir),
            plot_single_epoch_time(m, artifact_dir),
            plot_jit_warmup(m, artifact_dir),
        ]
        log_artifacts([p for p in paths if p])

        print_summary(m)

    return m


def parse_args():
    parser = argparse.ArgumentParser(description="Flax/JAX 실험")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_flax(args.config)