"""Scikit-learn MLP 모델 정의 (CIFAR-100)"""
from sklearn.neural_network import MLPClassifier


def build_sklearn_model(cfg: dict) -> MLPClassifier:
    """
    Scikit-learn MLPClassifier 생성.
    CNN이 불가능하므로 Flatten 후 FC 레이어로 구성.
    """
    train_cfg = cfg["train"]
    dense_units = cfg["model"]["dense_units"]  # list e.g. [512, 256]

    model = MLPClassifier(
        hidden_layer_sizes=tuple(dense_units),
        activation="relu",
        solver="adam",
        learning_rate_init=train_cfg["learning_rate"],
        max_iter=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        random_state=cfg["data"]["random_seed"],
        early_stopping=True,
        validation_fraction=cfg["data"]["val_split"],
        n_iter_no_change=10,
        verbose=False,
    )
    return model
