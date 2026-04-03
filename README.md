# Linen에서 NNX까지: JAX/Flax 고성능 파이프라인 구축 및 생산성 분석

본 프로젝트는 [CIFAR-10 이미지 분류 프로젝트](링크_추가예정)의 후속 연구로,
단순한 프레임워크 성능 비교를 넘어 **"실무에서 왜 JAX를 선택해야 하는가"** 라는 질문에
실험적 근거로 답합니다.
Scikit-learn → TensorFlow → Flax(Linen) 로 이어지는 흐름 속에서
각 프레임워크의 성능, 효율성, 유지보수성을 **MLflow** 기반으로 추적·시각화합니다.

---

## 1. 프로젝트 배경 (Background)

이전 CIFAR-10 프로젝트에서 TensorFlow 기반 Optimized CNN으로 **81.61%** 의 테스트 정확도를 달성했습니다.
이 경험을 바탕으로 다음 두 가지 질문을 탐구합니다:

- **"같은 모델을 다른 프레임워크로 구현하면 성능과 속도에 어떤 차이가 생기는가?"**
- **"JAX/Flax의 JIT 컴파일과 XLA 최적화는 컴퓨팅 효율성 측면에서 실제로 얼마나 효과적인가?"**

또한 CIFAR-10(10클래스) → **CIFAR-100(100클래스)** 로 데이터셋을 확장하여,
더 어려운 분류 문제에서 각 프레임워크의 한계와 강점이 명확히 드러나도록 설계했습니다.

---

## 2. 프로젝트 목표 (Objectives)

- 세 프레임워크(Scikit-learn / TensorFlow / Flax/JAX)로 **동일한 CNN 아키텍처** 구현
- JIT 컴파일 및 XLA 최적화 기반 **Flax/JAX의 컴퓨팅 효율성** 탐구
- MLflow를 활용한 실험 결과 **자동 추적 및 시각화**
- 정확도, 학습 시간, 메모리 사용량 등 **다각적 비교 지표** 제시
- MLOps 관점에서 **재현 가능한 실험 환경** 구축

---

## 3. 데이터셋 (Dataset)

| 항목 | 내용 |
|---|---|
| 데이터셋 | CIFAR-100 |
| 클래스 수 | 100개 (슈퍼클래스 20개) |
| 이미지 수 | 학습 50,000개 / 테스트 10,000개 |
| 이미지 크기 | 32×32 컬러(RGB) |
| 전처리 | 픽셀 정규화 (0~1), 학습-검증 분리 (8:2) |

> CIFAR-10 대비 클래스 수가 10배 많아 모델 수렴이 느리고,
> 프레임워크 간 학습 속도 및 정확도 차이가 더욱 뚜렷하게 나타납니다.

---

## 4. 프레임워크 구성 (Framework Setup)

| 프레임워크 | 모델 | MLflow 로깅 방식 |
|---|---|---|
| Scikit-learn | MLP (Flatten → FC layers) | `mlflow.sklearn.autolog()` |
| TensorFlow | CNN (Conv2D + MaxPool + Dense) | `mlflow.tensorflow.autolog()` |
| Flax/JAX (Linen) | CNN (동일 구조, Linen 모듈) | 수동 `mlflow.log_metric()` |

세 프레임워크 모두 **동일한 아키텍처와 하이퍼파라미터**를 적용하여 공정한 비교 환경을 구성합니다.

---

## 5. 프로젝트 구조 (Directory Structure)

```
ml-framework-comparison/
│
├── data/
│   └── download.py                # CIFAR-100 다운로드 스크립트
│
├── models/                        # 프레임워크별 모델 정의
│   ├── sklearn_model.py
│   ├── tensorflow_model.py
│   └── flax_model.py              # Linen 구현
│
├── experiments/                   # 실험 실행 스크립트
│   ├── run_sklearn.py
│   ├── run_tensorflow.py
│   └── run_flax.py
│
├── tracking/                      # MLflow 설정 및 공통 로거
│   ├── logger.py
│   └── config.py
│
├── utils/                         # 공통 유틸리티
│   ├── data_loader.py             # 공통 데이터 로더
│   ├── metrics.py                 # 공통 평가 지표
│   └── visualize.py               # 결과 시각화
│
├── notebooks/
│   └── comparison_demo.ipynb      # 세미나용 데모 노트북
│
├── mlruns/                        # MLflow 자동 생성 (gitignore)
├── config.yaml                    # 전역 하이퍼파라미터 설정
├── docker-compose.yml             # MLflow 서버 (포트 5001, macOS)
├── run_all.py                     # 전체 실험 일괄 실행
├── requirements.txt
└── README.md
```

---

## 6. 비교 지표 (Evaluation Metrics)

- **정확도 (Accuracy)**: Top-1 테스트 정확도
- **학습 시간**: 에폭당 평균 소요 시간 / JIT 워밍업 시간 별도 측정
- **처리량 (Throughput)**: 초당 처리 샘플 수 (samples/sec)
- **수렴 속도**: 목표 정확도 도달까지 필요한 에폭 수
- **메모리 사용량**: 학습 중 최대 메모리 점유율
- **MLflow 로깅 편의성**: 자동/수동 로깅 방식 비교

---

## 7. 실험 실행 방법 (Getting Started)

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 데이터 다운로드
python data/download.py

# 3. MLflow 서버 실행 (Docker)
docker-compose up -d
# → http://localhost:5001 에서 결과 확인

# 4. 전체 실험 실행
python run_all.py
```

> **Google Colab에서 실행 시**: ngrok으로 로컬 MLflow 서버와 연동합니다.
> ```python
> import yaml
> with open("config.yaml", "r") as f:
>     cfg = yaml.safe_load(f)
> cfg["mlflow"]["tracking_uri"] = "https://your-ngrok-address.ngrok-free.app"
> with open("config.yaml", "w") as f:
>     yaml.dump(cfg, f, allow_unicode=True)
> ```

---

## 8. 발표 목차 (Presentation Outline)

> 발표 시간: 15분 + Q&A 5분

| 섹션 | 내용 | 시간 |
|---|---|---|
| 1. 배경 및 동기 | CIFAR-10 회고, "왜 JAX인가?" | 1분 |
| 2. JAX란 무엇인가 | JIT 컴파일, XLA, NumPy 호환 API 소개 | 2분 |
| 3. 실험 설계 | CIFAR-100, 프레임워크 구성, MLflow | 2분 |
| 4. 성능 및 효율성 결과 | 정확도·속도 비교, JIT 워밍업, MLflow 시연 | 5분 |
| 5. 실무 시사점 | JAX 선택 기준, 컴퓨팅 효율성 분석 | 2분 |
| 6. 결론 | 요약 및 향후 방향 | 3분 |

---

## 9. 이전 프로젝트와의 연계 (Relation to Previous Work)

| 항목 | CIFAR-10 프로젝트 | 본 프로젝트 |
|---|---|---|
| 데이터셋 | CIFAR-10 (10클래스) | CIFAR-100 (100클래스) |
| 프레임워크 | TensorFlow 단일 | Scikit-learn / TF / Flax 비교 |
| 실험 관리 | 없음 | MLflow + Docker |
| 목적 | 모델 성능 최적화 | 프레임워크 실무 선택 근거 마련 |

---

## 10. 향후 계획 (Future Work)

- **Linen → NNX 전환**: 유지보수성 및 코드 복잡도 비교
- **Keras 3 백엔드 전환 실습**: TF ↔ JAX ↔ PyTorch 스위칭으로 마이그레이션 전략 검증
- **Orbax 체크포인팅**: 프로덕션 수준의 모델 저장·복원 파이프라인 구축
- **분산 학습**: JAX `pmap` 기반 멀티 GPU 학습 실험
- **PyTorch 프레임워크 추가 비교**
- 기술 블로그 포스팅 연계

---

**작성자:** 김동환