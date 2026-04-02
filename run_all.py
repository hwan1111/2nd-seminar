"""전체 실험 일괄 실행 스크립트"""
import argparse
import os
import sys

import yaml

from utils.visualize import plot_training_curves, plot_comparison_bar, save_results_csv


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-100 Framework Comparison")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["sklearn", "tensorflow", "flax"],
        choices=["sklearn", "tensorflow", "flax"],
        help="실행할 프레임워크 목록",
    )
    parser.add_argument("--skip-download", action="store_true", help="데이터 다운로드 건너뜀")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. 데이터 다운로드 (iris는 sklearn 내장 데이터라 불필요)
    if not args.skip_download and cfg["data"]["dataset"] != "iris":
        print("=" * 60)
        print("[STEP 1] CIFAR-100 데이터 준비")
        print("=" * 60)
        from data.download import download_cifar100
        download_cifar100(save_dir=cfg["data"]["data_dir"])

    # 2. 실험 실행
    metrics_dict = {}

    if "sklearn" in args.frameworks:
        print("\n" + "=" * 60)
        print("[STEP 2-1] Scikit-learn MLP 실험")
        print("=" * 60)
        from experiments.run_sklearn import run_sklearn
        metrics_dict["sklearn"] = run_sklearn(args.config)

    if "tensorflow" in args.frameworks:
        print("\n" + "=" * 60)
        print("[STEP 2-2] TensorFlow CNN 실험")
        print("=" * 60)
        from experiments.run_tensorflow import run_tensorflow
        metrics_dict["tensorflow"] = run_tensorflow(args.config)

    if "flax" in args.frameworks:
        print("\n" + "=" * 60)
        print("[STEP 2-3] Flax/JAX CNN 실험")
        print("=" * 60)
        from experiments.run_flax import run_flax
        metrics_dict["flax"] = run_flax(args.config)

    # 3. 결과 시각화 및 저장
    if metrics_dict:
        print("\n" + "=" * 60)
        print("[STEP 3] 결과 시각화 및 저장")
        print("=" * 60)
        results_dir = cfg.get("paths", {}).get("results_dir", "results")
        plot_training_curves(metrics_dict, save_dir=results_dir)
        plot_comparison_bar(metrics_dict, save_dir=results_dir)
        save_results_csv(metrics_dict, save_dir=results_dir)

        print("\n[완료] 모든 실험이 종료되었습니다.")
        print(f"  - MLflow UI: mlflow ui  → http://localhost:5000")
        print(f"  - 결과 파일: {results_dir}/")


if __name__ == "__main__":
    main()
