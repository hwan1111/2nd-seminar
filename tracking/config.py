"""MLflow 설정"""
import os
import yaml
import mlflow


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_mlflow(config_path: str = "config.yaml") -> str:
    """MLflow tracking URI 및 experiment 설정"""
    cfg = load_config(config_path)
    mlflow_cfg = cfg["mlflow"]

    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = mlflow_cfg.get("experiment_name", "cifar100-comparison")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    return experiment_name


def get_run_tags(framework: str, config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)
    return {
        "framework": framework,
        "dataset": cfg["data"]["dataset"],
        "num_classes": str(cfg["data"]["num_classes"]),
        "epochs": str(cfg["train"]["epochs"]),
        "batch_size": str(cfg["train"]["batch_size"]),
        "learning_rate": str(cfg["train"]["learning_rate"]),
    }
