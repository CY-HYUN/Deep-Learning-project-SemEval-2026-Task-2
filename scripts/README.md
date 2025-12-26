# Scripts 디렉토리 구조

**목적**: 0.70+ CCC 달성을 위한 스크립트 정리

---

## 📁 디렉토리 구조

```
scripts/
├── data_train/subtask2a/          # 모델 훈련 스크립트
│   ├── train_ensemble_subtask2a.py         # ⭐ 기본 훈련 (seed 변경)
│   ├── train_arousal_specialist.py         # ⭐ Arousal 전문 모델
│   └── train_arousal_focused.py            # (구버전, 사용 안함)
│
├── data_analysis/subtask2a/       # 분석 & 예측 & 검증 스크립트
│   ├── predict_test_subtask2a_optimized.py # ⭐ 최적화 예측 (2-model)
│   ├── calculate_optimal_ensemble_weights.py  # ⭐ 앙상블 조합 분석
│   ├── verify_test_data.py                 # ⭐ 평가 데이터 검증
│   ├── validate_predictions.py             # ⭐ 예측 결과 검증
│   ├── optimize_ensemble_stacking.py       # Stacking 최적화
│   └── predict_test_subtask2a.py           # (구버전, 사용 안함)
│
└── archive/                       # 완료/보관 스크립트
    ├── test_2model_ensemble.py             # 2-model 테스트 (완료 ✅)
    └── analyze_ensemble_weights_subtask2a.py  # (구버전)
```

---

## 🚀 사용 가이드

### Phase 1: 모델 훈련 (12/20-12/22)

#### 1. seed888 훈련 (Google Colab Pro)

```python
# train_ensemble_subtask2a.py 수정:
RANDOM_SEED = 888
MODEL_SAVE_NAME = 'subtask2a_seed888_best.pt'

# 실행
!python scripts/data_train/subtask2a/train_ensemble_subtask2a.py

# 결과:
# - models/subtask2a_seed888_best.pt
# - 예상 CCC: 0.58-0.63
```

#### 2. seed999 훈련 (조건부)

```python
# 조건: seed888 CCC ≥ 0.60

RANDOM_SEED = 999
MODEL_SAVE_NAME = 'subtask2a_seed999_best.pt'

!python scripts/data_train/subtask2a/train_ensemble_subtask2a.py
```

#### 3. Arousal Specialist 훈련 (핵심!)

```python
# train_arousal_specialist.py 사용
# 핵심 변경사항:
# - CCC_WEIGHT_A = 0.90 (Arousal 90%)
# - Arousal change features 추가
# - Weighted sampling

!python scripts/data_train/subtask2a/train_arousal_specialist.py

# 결과:
# - models/subtask2a_arousal_specialist_seed1111_best.pt
# - 예상 Arousal CCC: 0.60-0.65
```

---

### Phase 2: 앙상블 최적화 (12/22)

#### 4. 모든 모델 평가

```python
# calculate_optimal_ensemble_weights.py 업데이트
all_models = {
    "seed123": 0.5330,
    "seed777": 0.6554,
    "seed888": 0.XXXX,  # 훈련 후 CCC 입력
    "seed999": 0.XXXX,  # 훈련 후 CCC 입력
}

!python scripts/data_analysis/subtask2a/calculate_optimal_ensemble_weights.py

# 출력: results/subtask2a/optimal_ensemble.json
```

#### 5. Stacking 최적화 (고급)

```python
# 각 모델의 validation 예측 필요
# optimize_ensemble_stacking.py 사용

!python scripts/data_analysis/subtask2a/optimize_ensemble_stacking.py

# 출력:
# - results/subtask2a/stacking_optimization.json
# - Valence/Arousal 별도 가중치
```

---

### Phase 3: 평가파일 제출 (릴리스 후)

#### 6. 평가 데이터 검증

```bash
# Codabench에서 test_subtask2a.csv 다운로드 후
python scripts/data_analysis/subtask2a/verify_test_data.py

# 확인:
# - user_id, is_forecasting_user 컬럼
# - Forecasting users 수
# - 모든 users가 training data에 존재
```

#### 7. 예측 생성

```python
# predict_test_subtask2a_optimized.py 업데이트
# 최종 앙상블 가중치 입력:

# Valence
pred_valence = (
    0.65 * model777['valence'] +
    0.25 * model888['valence'] +
    0.10 * model123['valence']
)

# Arousal (Specialist 우선)
pred_arousal = (
    0.60 * arousal_specialist['arousal'] +
    0.30 * model777['arousal'] +
    0.10 * model888['arousal']
)

!python scripts/data_analysis/subtask2a/predict_test_subtask2a_optimized.py

# 출력: pred_subtask2a.csv
```

#### 8. 예측 검증 및 제출

```bash
# 1. 검증
python scripts/data_analysis/subtask2a/validate_predictions.py

# 확인:
# - 컬럼 정확성
# - NaN 값 없음
# - 모든 forecasting users 포함

# 2. 제출
# pred_subtask2a.csv → submission.zip
# Codabench 업로드
```

---

## 📊 스크립트 세부 정보

### 훈련 스크립트

#### `train_ensemble_subtask2a.py` ⭐⭐⭐
- **용도**: 기본 모델 훈련
- **변경**: RANDOM_SEED만 (42, 123, 777, 888, 999)
- **설정**:
  - CCC_WEIGHT_V = 0.65
  - CCC_WEIGHT_A = 0.70
  - SEQ_LENGTH = 7
  - BATCH_SIZE = 10
- **시간**: 1.5-2.5시간 (GPU 성능 따라)

#### `train_arousal_specialist.py` ⭐⭐⭐
- **용도**: Arousal 전문 모델
- **핵심 차이**:
  - CCC_WEIGHT_A = 0.90 (기존 0.70)
  - Arousal change features
  - Weighted sampling (Arousal 변화 큰 샘플 우선)
- **시간**: 2-4시간
- **예상 개선**: Arousal +0.05-0.10

#### `train_arousal_focused.py`
- **상태**: 구버전, 사용 안함
- **대체**: train_arousal_specialist.py

---

### 분석 스크립트

#### `test_2model_ensemble.py` ✅ (완료)
- **용도**: seed42 제거 효과 검증
- **결과**: CCC 0.5946 → 0.6305 (+6%)
- **결론**: seed42 제거 확정

#### `calculate_optimal_ensemble_weights.py` ⭐⭐⭐
- **용도**: 모든 모델 조합 중 최적 찾기
- **입력**: all_models dict (각 CCC)
- **출력**: optimal_ensemble.json
- **기능**:
  - 2-5 model 조합 테스트
  - CCC 기반 가중치
  - 앙상블 부스트 (+0.02-0.04)

#### `optimize_ensemble_stacking.py` ⭐⭐⭐ (고급)
- **용도**: Ridge Regression으로 최적 가중치 학습
- **필요**: 각 모델의 validation 예측
- **출력**: stacking_optimization.json
- **기능**:
  - Valence/Arousal 별도 최적화
  - 5-fold Cross Validation
  - Alpha grid search
- **예상 개선**: +0.01-0.02

#### `analyze_ensemble_weights_subtask2a.py`
- **상태**: 구버전
- **대체**: calculate_optimal_ensemble_weights.py

---

### 예측 스크립트

#### `predict_test_subtask2a_optimized.py` ⭐⭐⭐ (권장)
- **용도**: 최적화된 2-model 예측
- **모델**: seed123 + seed777 (seed42 제거)
- **가중치**: 성능 비례 (0.4485, 0.5515)
- **예상 CCC**: 0.6305

#### `predict_test_subtask2a.py`
- **용도**: 기본 3-model 예측
- **모델**: seed42 + seed123 + seed777
- **상태**: seed42 포함 (성능 낮음)
- **사용 안함**: seed42 제거 확정으로

---

### 검증 스크립트

#### `verify_test_data.py` ⭐⭐⭐
- **용도**: 평가 데이터 검증
- **확인 항목**:
  - 파일 로드 가능
  - 필수 컬럼 존재
  - Forecasting users 수
  - Training data와 일치
- **실행 시점**: 평가파일 다운로드 직후

#### `validate_predictions.py` ⭐⭐⭐
- **용도**: 예측 결과 검증
- **확인 항목**:
  - 컬럼명 정확성
  - 데이터 타입
  - NaN 값 없음
  - 중복 user_id 없음
  - 값 범위 합리성
  - 모든 forecasting users 포함
- **실행 시점**: 예측 생성 후, 제출 전

---

## 🎯 우선순위

### 필수 (Must Have) ⭐⭐⭐
1. `train_ensemble_subtask2a.py` - seed888 훈련
2. `train_arousal_specialist.py` - Arousal 전문 모델
3. `predict_test_subtask2a_optimized.py` - 최종 예측
4. `verify_test_data.py` - 데이터 검증
5. `validate_predictions.py` - 결과 검증

### 권장 (Should Have) ⭐⭐
6. `train_ensemble_subtask2a.py` - seed999 훈련
7. `calculate_optimal_ensemble_weights.py` - 조합 최적화
8. `optimize_ensemble_stacking.py` - Stacking 구현

### 선택 (Nice to Have) ⭐
9. Loss weight grid search
10. Sequence length variants
11. Additional seeds

---

## 📝 결과 파일

### `results/subtask2a/`

#### `ensemble_results.json` (기존)
```json
{
  "individual_models": {
    "seed42": {"ccc": 0.5053, ...},
    "seed123": {"ccc": 0.5330, ...},
    "seed777": {"ccc": 0.6554, ...}
  },
  "ensemble": {...}
}
```

#### `optimal_ensemble.json` (업데이트 필요)
```json
{
  "models": ["seed123", "seed777", "seed888"],
  "weights": {...},
  "ccc_avg": 0.6605
}
```

#### `stacking_optimization.json` (새로 생성)
```json
{
  "valence": {
    "weights": {"seed777": 0.65, "seed888": 0.25, ...},
    "ccc": 0.76
  },
  "arousal": {
    "weights": {"arousal_specialist": 0.60, "seed777": 0.30, ...},
    "ccc": 0.64
  }
}
```

#### `test_results_template.json`
- 제출 후 결과 기록용

---

## 🚀 Quick Start

### 시나리오 1: 기본 (8시간)
```bash
# Day 1
python scripts/data_train/subtask2a/train_ensemble_subtask2a.py  # seed888
python scripts/data_train/subtask2a/train_arousal_specialist.py

# Day 2
python scripts/data_analysis/subtask2a/calculate_optimal_ensemble_weights.py
# 예측 스크립트 가중치 업데이트
# 평가파일 대기
```

### 시나리오 2: 고급 (12시간)
```bash
# 기본 + Stacking
python scripts/data_train/subtask2a/train_ensemble_subtask2a.py  # seed999도
python scripts/data_analysis/subtask2a/optimize_ensemble_stacking.py
# Ridge regression 가중치 적용
```

---

**마지막 업데이트**: 2025-12-24
**현재 상태**: ✅ 모든 모델 훈련 완료, 평가파일 대기 중
**최종 성능**: CCC 0.6833 (목표 0.62 대비 +10.4%)
