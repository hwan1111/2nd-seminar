"""전체 실험 일괄 실행 스크립트"""
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="ML Framework Comparison")
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

    # 데이터 다운로드
    if not args.skip_download and cfg["data"]["dataset"] != "iris":
        print("=" * 60)
        print("[STEP 1] 데이터 준비")
        print("=" * 60)
        from data.download import download_cifar100
        download_cifar100(save_dir=cfg["data"]["data_dir"])

    # 실험 실행 — 시각화/artifact 저장은 각 run_*.py에서 처리
    runners = {
        "sklearn":    ("experiments.run_sklearn",    "run_sklearn"),
        "tensorflow": ("experiments.run_tensorflow", "run_tensorflow"),
        "flax":       ("experiments.run_flax",       "run_flax"),
    }

    for fw in args.frameworks:
        print("\n" + "=" * 60)
        print(f"[실험] {fw}")
        print("=" * 60)
        module_path, func_name = runners[fw]
        import importlib
        module = importlib.import_module(module_path)
        run_fn = getattr(module, func_name)
        run_fn(args.config)

    print("\n[완료] 모든 실험이 종료되었습니다.")
    print(f"  - MLflow UI: {cfg['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    main()