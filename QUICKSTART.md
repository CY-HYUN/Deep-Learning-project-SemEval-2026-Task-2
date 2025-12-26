# 🚀 즉시 시작 가이드

**현재 상태**: CCC 0.6305 (목표 초과 ✅)
**최종 목표**: CCC 0.70-0.72
**필요 시간**: 주말 8시간

---

## ✅ 완료된 작업

- ✅ 설문조사 작성
- ✅ Zoom 건너뜀 (OK)
- ✅ seed42 제거 (Arousal 낮음)
- ✅ 2-model baseline (CCC 0.6305)

---

## 📋 할 일 (순서대로)

### 1단계: seed888 훈련 (2시간) - 지금!

#### Google Colab Pro 준비
```
1. https://colab.research.google.com/ 열기
2. 런타임 → 런타임 유형 변경 → GPU (A100 > V100 > T4)
3. GPU 확인: !nvidia-smi
```

#### 파일 업로드
```python
from google.colab import files
uploaded = files.upload()

# 업로드할 파일:
# 1) D:\Study\Github\Deep-Learning-project-SemEval-2026-Task-2\scripts\data_train\subtask2a\train_ensemble_subtask2a.py
# 2) D:\Study\Github\Deep-Learning-project-SemEval-2026-Task-2\data\train_subtask2a.csv
```

#### 스크립트 수정 (2줄만!)
```
파일: train_ensemble_subtask2a.py

Line 29: RANDOM_SEED = 888  (777 → 888)
Line 30: MODEL_SAVE_NAME = 'subtask2a_seed888_best.pt'
Line 33: USE_WANDB = False  (확인)
Line 96-98: # 주석 처리 (파일 업로드 라인)
```

#### 실행
```python
!python train_ensemble_subtask2a.py
# 2-2.5시간 대기
```

#### 결과 확인
```
Best Validation CCC: 0.XXXX  ← 기록!
Valence: 0.XXXX
Arousal: 0.XXXX
```

#### 다운로드
```python
files.download('subtask2a_seed888_best.pt')
# 저장: D:\Study\Github\Deep-Learning-project-SemEval-2026-Task-2\models\
```

**판단**:
- CCC ≥ 0.60: ✅ 2단계로
- CCC < 0.60: ⚠️ 3단계로 (seed999 건너뛰기)

---

### 2단계: Arousal Specialist 훈련 (4시간) ⭐ 핵심!

#### 왜 중요?
```
현재: Arousal 0.55 << Valence 0.76 (27% 차이)
해결: Arousal 전문 모델
예상: +0.05-0.08 개선 (가장 큰 효과!)
```

#### 파일 복사
```
train_ensemble_subtask2a.py 복사
→ train_arousal_specialist.py 생성
```

#### 7가지 핵심 수정

**1. Seed & 파일명** (Line 29-30):
```python
RANDOM_SEED = 1111
MODEL_SAVE_NAME = 'subtask2a_arousal_specialist_seed1111_best.pt'
```

**2. Loss Weights** (Line 248-251) ⭐⭐⭐:
```python
CCC_WEIGHT_V = 0.50  # Valence 보조
CCC_WEIGHT_A = 0.90  # ⭐ Arousal 집중! (70% → 90%)
MSE_WEIGHT_V = 0.50
MSE_WEIGHT_A = 0.10  # ⭐ (30% → 10%)
```

**3. Arousal Features 추가** (Line 188 다음):
```python
# ===== AROUSAL SPECIALIST FEATURES =====
df['arousal_change'] = df.groupby('user_id')['arousal'].diff().abs().fillna(0)
df['arousal_volatility'] = df.groupby('user_id')['arousal'].transform(
    lambda x: x.rolling(5, min_periods=1).std()
).fillna(0)
df['arousal_acceleration'] = df.groupby('user_id')['arousal_change'].diff().fillna(0)
```

**4. Dataset 수정** (Line 327-332):
```python
temp_features = seq_data[[
    'valence_lag1', 'valence_lag2', 'valence_lag3', 'valence_lag4', 'valence_lag5',
    'arousal_lag1', 'arousal_lag2', 'arousal_lag3', 'arousal_lag4', 'arousal_lag5',
    'time_gap_log', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'entry_number', 'relative_position',
    'arousal_change', 'arousal_volatility', 'arousal_acceleration'  # ⭐ 추가
]].values.astype(np.float32)
```

**5. Input Dimension** (Line 378):
```python
temp_feature_dim = 20  # ⭐ 17 → 20
```

**6. Weighted Sampling** (Line 363):
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

**7. WandB & 업로드** (Line 33, 96-98):
```python
USE_WANDB = False  # 확인
# 파일 업로드 라인 주석 처리
```

#### 실행
```python
# Google Colab Pro 업로드
!python train_arousal_specialist.py
# 4-5시간 대기
```

#### 결과 확인 (Arousal CCC만 중요!)
```
Best Arousal CCC: 0.XXXX  ← 0.60+ 이면 성공!
```

#### 다운로드
```python
files.download('subtask2a_arousal_specialist_seed1111_best.pt')
```

---

### 3단계: seed999 훈련 (2시간, 선택)

**조건**: seed888 CCC ≥ 0.60

seed888과 동일, RANDOM_SEED만 변경:
```python
RANDOM_SEED = 999
MODEL_SAVE_NAME = 'subtask2a_seed999_best.pt'
```

---

### 4단계: 최종 앙상블 구성 (1시간)

#### 모든 모델 CCC 기록
```
seed123: 0.5330 ✅
seed777: 0.6554 ✅
seed888: 0.XXXX (1단계 결과)
arousal_specialist: Arousal 0.XXXX (2단계 결과)
seed999: 0.XXXX (3단계, 선택)
```

#### 가중치 계산
```bash
# 로컬에서 실행
python scripts/data_analysis/subtask2a/calculate_optimal_ensemble_weights.py
```

#### 예측 스크립트 업데이트
파일: `scripts/data_analysis/subtask2a/predict_test_subtask2a_optimized.py`

```python
# Valence: 기존 모델 우선
pred_valence = (
    0.60 * model777['valence'] +
    0.25 * model888['valence'] +
    0.15 * model123['valence']
)

# Arousal: Specialist 우선
pred_arousal = (
    0.60 * arousal_specialist['arousal'] +
    0.30 * model777['arousal'] +
    0.10 * model888['arousal']
)
```

---

### 5단계: 평가파일 릴리스 대기 (12/23-25)

**모니터링**: https://www.codabench.org/competitions/9963/

---

### 6단계: 제출 (1시간)

#### 다운로드 & 검증
```bash
# Codabench → Files → test_subtask2a.csv 다운로드
# 저장: data/test/

python scripts/verify_test_data.py
```

#### 예측 생성 (Google Colab Pro 추천)
```python
# 모든 모델 파일 업로드
# predict_test_subtask2a_optimized.py 업로드

!python predict_test_subtask2a_optimized.py
# 결과: pred_subtask2a.csv
```

#### 검증 & 제출
```bash
python scripts/validate_predictions.py

# pred_subtask2a.csv → submission.zip
# Codabench 업로드
```

---

## 📊 예상 결과

### Conservative (85% 확률)
```
seed888 + Arousal Specialist
→ CCC 0.68-0.70
```

### Aggressive (70% 확률)
```
+ seed999
→ CCC 0.70-0.72
```

---

## 📅 타임라인

**오늘 (12/19)**: seed888 시작
**내일 (12/20)**: Arousal Specialist
**모레 (12/21)**: seed999 (선택) + 앙상블
**평가파일 후**: 제출 (1시간)

---

## 💡 핵심 포인트

1. **seed888**: 빠르게 시작 (2시간)
2. **Arousal Specialist**: 가장 큰 개선 (+0.05-0.08) ⭐⭐⭐
3. **seed999**: 선택사항 (+0.01)

---

## 🆘 Troubleshooting

### GPU 메모리 부족
```python
BATCH_SIZE = 8  # 10 → 8
```

### Arousal CCC 안 올라감
```python
CCC_WEIGHT_A = 0.95  # 90% → 95%
```

---

## 📞 빠른 참조

- **스크립트**: `scripts/data_train/subtask2a/train_ensemble_subtask2a.py`
- **데이터**: `data/train_subtask2a.csv`
- **모델 저장**: `models/`
- **Codabench**: https://www.codabench.org/competitions/9963/

---

🚀 **지금 바로 1단계 시작!**
