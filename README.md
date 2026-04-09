# 왜 지금 JAX인가: 프레임워크 선택이 성능이 아니라 컴퓨팅 전략이 되는 순간

본 프로젝트는 동일한 딥러닝 모델을 서로 다른 프레임워크로 구현하여  
**프레임워크 선택이 실제 성능, 효율성, 확장성에 어떤 영향을 미치는지**를 검증하는 것을 목표로 합니다.

단순한 성능 비교를 넘어,  
JAX가 가지는 컴파일 기반 실행 모델(JIT + XLA)이  
실무 환경에서 어떤 의미를 가지는지 실험적으로 분석합니다.

---

## 1. Problem Statement

딥러닝 모델의 성능은 동일하더라도,  
프레임워크에 따라 학습 속도, 메모리 사용량, 확장성이 크게 달라질 수 있습니다.

그러나 대부분의 프로젝트에서는 프레임워크 선택이  
익숙함이나 생태계에 기반해 이루어지는 경우가 많으며,  
컴퓨팅 효율성 관점에서의 정량적 비교는 상대적으로 부족합니다.

본 프로젝트는 다음 질문을 검증합니다:

> 동일한 모델을 서로 다른 프레임워크로 구현했을 때,  
> 실제 성능 및 효율성 차이는 어떻게 나타나는가?

---

## 2. Why JAX?

JAX는 단순한 딥러닝 프레임워크가 아니라,  
**컴파일 기반 실행 모델을 채택한 시스템**입니다.

기존 프레임워크(TensorFlow, PyTorch)는 eager execution을 기반으로 하는 반면,  
JAX는 JIT 컴파일을 통해 연산 그래프를 최적화합니다.

이러한 구조적 차이는 다음과 같은 가설을 가능하게 합니다:

- 동일한 모델에서도 더 높은 처리량(throughput)을 달성할 수 있다  
- 연산 최적화를 통해 메모리 효율성이 개선될 수 있다  
- 초기 컴파일 비용(JIT warmup) 이후 성능이 안정적으로 향상될 수 있다  

본 프로젝트는 이러한 가설을 실험적으로 검증합니다.

---

## 3. Approach

본 프로젝트는 **프레임워크 자체의 차이만을 비교**하기 위해,  
다음과 같은 설계 원칙을 적용합니다:

- 동일한 데이터셋 사용 (CIFAR-100)
- 유사한 CNN 아키텍처 유지
- 동일한 하이퍼파라미터 적용
- 동일한 학습 조건에서 실험 수행

이를 통해 프레임워크 외의 변수를 최대한 통제하고,  
구조적 차이에 따른 성능 변화를 관찰합니다.

---

## 4. Experiment Design

### Dataset

| 항목 | 내용 |
|---|---|
| 데이터셋 | CIFAR-100 |
| 클래스 수 | 100 |
| 학습 데이터 | 50,000 |
| 테스트 데이터 | 10,000 |
| 이미지 크기 | 32×32 RGB |
| 전처리 | 정규화 (0~1), train/validation split (8:2) |

---

### Framework Setup

| 프레임워크 | 모델 | 로깅 방식 |
|---|---|---|
| Scikit-learn | MLP |`mlflow.sklearn.autolog()`|
| TensorFlow | CNN | `mlflow.tensorflow.autolog()` |
| Flax/JAX | CNN | 수동 `mlflow.log_metric()` |

세 프레임워크 모두 **유사한 구조의 모델**을 사용하여  
공정한 비교 환경을 구성합니다.

---

### Evaluation Metrics

- Accuracy (Top-1)
- Training Time (epoch당 시간)
- Throughput (samples/sec)
- Convergence Speed (목표 정확도 도달 epoch)
- Memory Usage
- Logging Complexity (MLflow 기준)

---

## 5. Expected Results

본 실험은 다음과 같은 결과를 검증하는 것을 목표로 합니다:

- JAX는 초기 JIT 컴파일 오버헤드가 존재하지만,  
  이후 학습에서는 더 높은 처리량을 보일 가능성이 있음

- 동일 정확도 기준에서  
  학습 시간 단축이 가능한지 확인

- 프레임워크별 메모리 사용 패턴 차이 분석

본 프로젝트는 단순 성능 비교가 아니라,  
**실행 모델 차이에 따른 구조적 효율성**을 확인하는 데 목적이 있습니다.

---

## 6. Project Structure
```
ml-framework-comparison/
│
├── data/
│ └── download.py
│
├── models/
│ ├── sklearn_model.py
│ ├── tensorflow_model.py
│ └── flax_model.py
│
├── experiments/
│ ├── run_sklearn.py
│ ├── run_tensorflow.py
│ └── run_flax.py
│
├── tracking/
│ ├── logger.py
│ └── config.py
│
├── utils/
│ ├── data_loader.py
│ ├── metrics.py
│ └── visualize.py
│
├── notebooks/
│ └── comparison_demo.ipynb
│
├── config.yaml
├── docker-compose.yml
├── run_all.py
└── requirements.txt
```

---

## 7. Tech Stack & Decisions

- **JAX / Flax**  
  → 컴파일 기반 실행 모델 검증

- **TensorFlow**  
  → 산업 표준 프레임워크 기준선

- **Scikit-learn**  
  → 단순 모델 baseline

- **MLflow**  
  → 실험 추적 및 재현성 확보

- **Docker**  
  → 실행 환경 일관성 유지

```bash
ngrok http 5001 --host-header="localhost:5001"
```
```python
mlflow.set_tracking_uri("https://temperamental-kazuko-enamouredly.ngrok-free.dev")
```
---

## 8. Limitations

- GPU 환경 기반 실험 (TPU 미포함)
- 단순 CNN 구조로 제한된 실험
- PyTorch 비교 미포함
- 데이터셋 규모 제한 (CIFAR-100)

---

## 9. Future Work

- PyTorch 프레임워크 추가 비교
- JAX 분산 학습 (pmap) 실험
- Keras 3 backend 전환 실습 (TF ↔ JAX)
- TPU / Trainium 환경에서의 확장 실험
- 대규모 데이터셋 기반 검증

---

## 10. Conclusion

본 프로젝트는 단순한 프레임워크 비교를 넘어,  
**실행 방식(eager vs compiled)이 성능과 효율성에 미치는 영향**을 분석합니다.

이를 통해 JAX가 단순한 선택지가 아니라,  
**컴퓨팅 전략 관점에서 고려해야 할 기술**임을 검증하는 것을 목표로 합니다.

---

**작성자**: 김동환  