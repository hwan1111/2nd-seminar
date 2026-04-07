# ============================================================
# models/model_registry.py
# config.yaml의 module명으로 동적 import
# ============================================================

import importlib


def get_flax_builder(cfg: dict):
    module_name = f"models.{cfg['model']['flax_module']}"
    module = importlib.import_module(module_name)
    return module.build_flax_model, module.cross_entropy_loss, module.compute_accuracy


def get_tensorflow_builder(cfg: dict):
    module_name = f"models.{cfg['model']['tensorflow_module']}"
    module = importlib.import_module(module_name)
    return module.build_tensorflow_model


def get_sklearn_builder(cfg: dict):
    module_name = f"models.{cfg['model']['sklearn_module']}"
    module = importlib.import_module(module_name)
    return module.build_sklearn_model