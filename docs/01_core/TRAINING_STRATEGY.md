# 훈련 전략 가이드

**작성일**: 2025-12-19
**목적**: 성능 향상을 위한 체계적 전략

---

## 🎯 현재 상태 → 목표

### 현재
```
CCC: 0.6305 (목표 0.62 초과 ✅)
Valence: 0.76 (좋음)
Arousal: 0.55 (개선 필요)
```

### 목표 범위
```
Conservative (85%): 0.68-0.70
Aggressive (70%): 0.70-0.72
```

---

## 📊 성능 향상 로드맵

| 단계 | 전략 | 예상 개선 | 누적 CCC | 시간 | 우선순위 | 성공률 |
|------|------|-----------|----------|------|----------|--------|
| 현재 | 2-model | - | 0.6305 | - | - | 100% |
| 1단계 | seed888 추가 | +0.015 | 0.6455 | 2h | ⭐⭐⭐ | 70% |
| 2단계 | Arousal Specialist | +0.050 | 0.6955 | 4h | ⭐⭐⭐ | 70% |
| 3단계 | Stacking 최적화 | +0.010 | 0.7055 | 2h | ⭐⭐ | 80% |
| 4단계 | seed999 추가 | +0.010 | 0.7155 | 2h | ⭐ | 60% |

**최소 목표** (1-2단계): CCC 0.70 (6시간, 85% 확률)
**최대 목표** (1-4단계): CCC 0.72 (10시간, 60% 확률)

---

## 🚀 1단계: seed888 훈련

### 목표
- CCC 0.60-0.63 달성
- seed777 패턴 재현

### 실행 방법

#### Google Colab Pro 설정
```python
# 1. Colab 열기 및 GPU 설정
런타임 → 런타임 유형 변경 → GPU (A100 > V100 > T4)

# 2. GPU 확인
!nvidia-smi
```

#### 파일 업로드
```python
from google.colab import files

# 필요한 파일:
# 1. train_ensemble_subtask2a.py
# 2. train_subtask2a.csv
uploaded = files.upload()
```

#### 스크립트 수정 (2줄만!)
```python
# train_ensemble_subtask2a.py
RANDOM_SEED = 888  # 777 → 888
MODEL_SAVE_NAME = 'subtask2a_seed888_best.pt'
USE_WANDB = False  # 확인
```

#### 실행
```bash
!python train_ensemble_subtask2a.py
# 예상 시간: 2-2.5시간
```

#### 결과 확인
```
Best Validation CCC: 0.XXXX
Valence CCC: 0.XXXX
Arousal CCC: 0.XXXX
```

#### 다운로드
```python
files.download('subtask2a_seed888_best.pt')
# 저장: models/subtask2a_seed888_best.pt
```

### 판단 기준
```
CCC ≥ 0.60: ✅ 성공 → 2단계 진행
CCC 0.58-0.60: ⚠️ 사용 가능 → 2단계 진행
CCC < 0.58: ❌ 실패 → 2-model 유지, 2단계로 건너뛰기
```

---

## 🎯 2단계: Arousal Specialist 훈련

### 목표
- Arousal CCC 0.60+ 달성
- 전체 CCC +0.05-0.08 개선

### 왜 중요한가?
```
현재 문제:
- Valence: 0.76 (좋음) ✅
- Arousal: 0.55 (낮음) ⚠️
- 차이: 27%

Arousal Specialist가 이 문제를 직접 해결!
```

### 핵심 수정사항 (7가지)

#### 1. Seed & 파일명 (Line 29-30)
```python
RANDOM_SEED = 1111
MODEL_SAVE_NAME = 'subtask2a_arousal_specialist_seed1111_best.pt'
```

#### 2. Loss Weights (Line 248-251) ⭐⭐⭐
```python
CCC_WEIGHT_V = 0.50  # Valence 보조 (기존: 0.65)
CCC_WEIGHT_A = 0.90  # ⭐ Arousal 집중! (기존: 0.70)
MSE_WEIGHT_V = 0.50  # (기존: 0.35)
MSE_WEIGHT_A = 0.10  # ⭐ (기존: 0.30)
```

#### 3. Arousal Features 추가 (Line 188 다음)
```python
# ===== AROUSAL SPECIALIST FEATURES =====
df['arousal_change'] = df.groupby('user_id')['arousal'].diff().abs().fillna(0)
df['arousal_volatility'] = df.groupby('user_id')['arousal'].transform(
    lambda x: x.rolling(5, min_periods=1).std()
).fillna(0)
df['arousal_acceleration'] = df.groupby('user_id')['arousal_change'].diff().fillna(0)
```

#### 4. Dataset 수정 (Line 327-332)
```python
temp_features = seq_data[[
    'valence_lag1', 'valence_lag2', 'valence_lag3', 'valence_lag4', 'valence_lag5',
    'arousal_lag1', 'arousal_lag2', 'arousal_lag3', 'arousal_lag4', 'arousal_lag5',
    'time_gap_log', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'entry_number', 'relative_position',
    'arousal_change', 'arousal_volatility', 'arousal_acceleration'  # ⭐ 추가
]].values.astype(np.float32)
```

#### 5. Input Dimension (Line 378)
```python
temp_feature_dim = 20  # ⭐ 17 → 20
```

#### 6. Weighted Sampling (Line 363)
```python
from torch.utils.data import WeightedRandomSampler

train_indices = train_df.index.tolist()
sample_weights = (train_df.loc[train_indices, 'arousal_change'] + 0.5).values
sample_weights = sample_weights / sample_weights.sum()

train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,  # ⭐ shuffle=True 대신
    num_workers=2
)
```

#### 7. WandB 비활성화
```python
USE_WANDB = False
# 파일 업로드 라인 주석 처리 (Line 96-98)
```

### 실행
```bash
# Google Colab Pro
!python train_arousal_specialist.py
# 예상 시간: 4-5시간
```

### 결과 확인 (Arousal CCC만 중요!)
```
Best Arousal CCC: 0.XXXX  ← 0.60+ 이면 성공!
```

### 다운로드
```python
files.download('subtask2a_arousal_specialist_seed1111_best.pt')
```

---

## 🔄 3단계: Stacking 최적화

### 목표
- Valence/Arousal 별도 가중치 최적화
- +0.01-0.02 개선

### 방법

#### 1. Validation 예측 저장
```python
# train_ensemble_subtask2a.py의 validation loop에 추가

if val_ccc > best_val_ccc:
    # 기존 모델 저장
    torch.save(...)

    # ⭐ 예측 저장 (새로 추가)
    val_predictions = {
        'valence': all_val_preds_v.cpu().numpy(),
        'arousal': all_val_preds_a.cpu().numpy(),
        'true_valence': all_val_labels_v.cpu().numpy(),
        'true_arousal': all_val_labels_a.cpu().numpy()
    }
    np.save(f'results/subtask2a/val_preds_seed{RANDOM_SEED}.npy',
            val_predictions)
```

#### 2. 최적화 실행
```bash
python scripts/data_analysis/subtask2a/optimize_ensemble_stacking.py
```

#### 3. 결과 확인
```json
{
  "baseline_ccc": 0.6955,
  "optimized_ccc": 0.7055,
  "improvement": +0.01,
  "valence_weights": {
    "seed777": 0.65,
    "seed888": 0.25,
    "arousal_specialist": 0.10
  },
  "arousal_weights": {
    "arousal_specialist": 0.70,
    "seed777": 0.20,
    "seed888": 0.10
  }
}
```

---

## 🎲 4단계: seed999 훈련 (조건부)

### 조건
- seed888 CCC ≥ 0.60

### 방법
seed888과 동일, RANDOM_SEED만 변경:
```python
RANDOM_SEED = 999
MODEL_SAVE_NAME = 'subtask2a_seed999_best.pt'
```

### 시간
2시간

### 예상 개선
+0.005-0.01 (미미함)

---

## 📈 예상 결과 분석

### Scenario 1: seed888만 추가
```
모델: seed123 + seed777 + seed888
시간: 2시간
성공률: 70%
예상 CCC: 0.63-0.65
```

### Scenario 2: seed888 + Arousal Specialist
```
모델: seed123 + seed777 + seed888 + arousal_specialist
시간: 6시간
성공률: 50%
예상 CCC: 0.68-0.72
```

### Scenario 3: Full (+ Stacking)
```
모델: 위 + Stacking 최적화
시간: 8시간
성공률: 40%
예상 CCC: 0.69-0.73
```

### Scenario 4: Maximum (+ seed999)
```
모델: 위 + seed999
시간: 10시간
성공률: 30%
예상 CCC: 0.70-0.74
```

---

## 💡 추천 전략

### Option A: 보수적 (추천) ⭐⭐⭐
```
단계: 1단계만 (seed888)
시간: 2시간
성공률: 70%
예상 CCC: 0.63-0.65
리스크: 매우 낮음
```

**이유**:
- 현재 이미 목표 달성 (0.6305)
- 낮은 시간 투자
- 실패해도 현재 유지

### Option B: 공격적
```
단계: 1-2단계 (seed888 + Arousal Specialist)
시간: 6시간
성공률: 50%
예상 CCC: 0.68-0.72
리스크: 중간
```

**이유**:
- Arousal Specialist가 가장 큰 개선
- 목표 0.70 달성 가능
- 충분한 시간 투자 가치

### Option C: 최대
```
단계: 1-4단계 전부
시간: 10시간
성공률: 30%
예상 CCC: 0.70-0.74
리스크: 높음
```

**이유**:
- 최고 성능 추구
- 시간 대비 효율 낮음
- 3-4단계 개선 미미

---

## ⚠️ 주의사항

### GPU 메모리 부족 시
```python
BATCH_SIZE = 8  # 10 → 8
# 또는
BATCH_SIZE = 4
```

### Arousal CCC 안 올라갈 시
```python
CCC_WEIGHT_A = 0.95  # 90% → 95%
MSE_WEIGHT_A = 0.05
```

### 훈련 중단 시
```python
# 체크포인트 로드
checkpoint = torch.load('last_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 📊 성공 지표

### 1단계 성공
```
seed888 CCC ≥ 0.60
3-model CCC ≥ 0.63
```

### 2단계 성공
```
Arousal Specialist Arousal CCC ≥ 0.60
전체 CCC ≥ 0.68
```

### 3단계 성공
```
Stacking 최적화 후 CCC ≥ 0.70
```

### 최종 성공
```
전체 CCC ≥ 0.70
Valence CCC ≥ 0.75
Arousal CCC ≥ 0.60
```

---

## 🔗 관련 문서

- **[QUICKSTART.md](../QUICKSTART.md)**: 즉시 실행 가이드
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)**: 현재 상태
- **[README.md](../README.md)**: 프로젝트 개요

---

**다음 단계**: seed888 훈련 시작 (선택)
**최종 목표**: CCC 0.70+ 달성
