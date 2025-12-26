# Results 디렉토리 - Subtask 2a

**목적**: 모델 성능 및 앙상블 결과 추적

---

## 📊 현재 파일

### `ensemble_results.json` (기존)

**내용**: 3-model 앙상블 결과 (seed42 포함)

```json
{
  "individual_models": {
    "seed42": {
      "ccc": 0.5053,
      "valence_ccc": 0.6532,
      "arousal_ccc": 0.3574,  // ⚠️ 매우 낮음
      "epoch": 16
    },
    "seed123": {
      "ccc": 0.5330,
      "valence_ccc": 0.6298,
      "arousal_ccc": 0.4362,
      "epoch": 18
    },
    "seed777": {
      "ccc": 0.6554,  // ⭐ 최고
      "valence_ccc": 0.7593,
      "arousal_ccc": 0.5516,
      "epoch": 9
    }
  },
  "ensemble": {
    "weights": {
      "seed42": 0.2983,
      "seed123": 0.3147,
      "seed777": 0.3870
    },
    "expected_ccc_min": 0.5846,
    "expected_ccc_max": 0.6046
  }
}
```

**상태**: 참고용 (seed42 제거로 더 이상 사용 안함)

---

### `optimal_ensemble.json` (업데이트됨)

**내용**: seed42 제거 후 최적 조합

```json
{
  "models": ["seed123", "seed777"],
  "weights": {
    "seed123": 0.4485,
    "seed777": 0.5515
  },
  "ccc_min": 0.6205,
  "ccc_max": 0.6405,
  "ccc_avg": 0.6305  // ✅ 목표 0.62 초과!
}
```

**상태**: 현재 Baseline

**업데이트 계획**:
```json
// seed888, seed999 훈련 후:
{
  "models": ["seed123", "seed777", "seed888", "seed999"],
  "weights": {
    "seed123": 0.15,
    "seed777": 0.40,
    "seed888": 0.25,
    "seed999": 0.20
  },
  "ccc_avg": 0.6605  // 예상
}
```

---

### `test_results_template.json`

**목적**: 제출 후 결과 기록용

```json
{
  "submission_date": "2025-12-XX",
  "models_used": ["seed123", "seed777", "seed888", "arousal_specialist"],
  "ensemble_strategy": "stacking with separate valence/arousal weights",
  "validation_ccc": 0.70,
  "test_results": {
    "pearson_r_valence": null,  // 제출 후 업데이트
    "pearson_r_arousal": null,
    "mae_valence": null
  },
  "ranking": {
    "position": null,
    "total_teams": null
  }
}
```

---

## 🎯 앞으로 생성될 파일

### `stacking_optimization.json` (새로 생성 예정)

**목적**: Stacking 최적화 결과

**예상 내용**:
```json
{
  "method": "ridge_regression",
  "baseline": {
    "method": "simple_weighted_average",
    "ccc": 0.6505,
    "weights": {
      "seed123": 0.20,
      "seed777": 0.45,
      "seed888": 0.35
    }
  },
  "optimized": {
    "ccc": 0.6655,
    "improvement": 0.015,
    "valence": {
      "ccc": 0.7650,
      "alpha": 0.1,
      "weights": {
        "seed777": 0.65,  // Valence 강함
        "seed888": 0.25,
        "seed123": 0.10
      }
    },
    "arousal": {
      "ccc": 0.6400,
      "alpha": 0.5,
      "weights": {
        "arousal_specialist": 0.60,  // Arousal 전문
        "seed777": 0.30,
        "seed888": 0.10
      }
    }
  }
}
```

**생성 방법**:
```bash
python scripts/data_analysis/subtask2a/optimize_ensemble_stacking.py
```

---

### `model_training_log.json` (새로 생성 권장)

**목적**: 모든 훈련 모델 추적

**예상 내용**:
```json
{
  "models": [
    {
      "name": "seed123",
      "ccc": 0.5330,
      "valence_ccc": 0.6298,
      "arousal_ccc": 0.4362,
      "trained_date": "2025-11-20",
      "epochs": 18,
      "status": "active"
    },
    {
      "name": "seed777",
      "ccc": 0.6554,
      "valence_ccc": 0.7593,
      "arousal_ccc": 0.5516,
      "trained_date": "2025-11-20",
      "epochs": 9,
      "status": "active",
      "notes": "Best individual model"
    },
    {
      "name": "seed42",
      "ccc": 0.5053,
      "valence_ccc": 0.6532,
      "arousal_ccc": 0.3574,
      "trained_date": "2025-11-20",
      "epochs": 16,
      "status": "removed",
      "notes": "Low Arousal performance, removed from ensemble"
    },
    {
      "name": "seed888",
      "ccc": null,  // 훈련 후 업데이트
      "valence_ccc": null,
      "arousal_ccc": null,
      "trained_date": "2025-12-21",
      "epochs": null,
      "status": "planned",
      "expected_ccc": "0.60-0.63"
    },
    {
      "name": "seed999",
      "ccc": null,
      "trained_date": null,
      "status": "conditional",
      "condition": "seed888 CCC >= 0.60"
    },
    {
      "name": "arousal_specialist_seed1111",
      "ccc": null,
      "valence_ccc": null,
      "arousal_ccc": null,  // 목표: 0.60-0.65
      "trained_date": "2025-12-21",
      "status": "planned",
      "notes": "Arousal-focused model with CCC_WEIGHT_A=0.90"
    }
  ],
  "ensemble_history": [
    {
      "date": "2025-11-20",
      "models": ["seed42", "seed123", "seed777"],
      "ccc": 0.6021,
      "status": "deprecated"
    },
    {
      "date": "2025-12-19",
      "models": ["seed123", "seed777"],
      "ccc": 0.6305,
      "status": "baseline",
      "notes": "seed42 removed, +6% improvement"
    },
    {
      "date": "2025-12-22",
      "models": ["seed123", "seed777", "seed888", "arousal_specialist"],
      "ccc": null,  // 예상 0.70
      "status": "planned"
    }
  ]
}
```

---

### `validation_predictions/` (새 디렉토리)

**목적**: Stacking 최적화를 위한 validation 예측 저장

**파일 구조**:
```
validation_predictions/
├── val_preds_seed123.npy
├── val_preds_seed777.npy
├── val_preds_seed888.npy
├── val_preds_seed999.npy
└── val_preds_arousal_specialist.npy
```

**각 파일 내용**:
```python
{
    'valence': np.array([...]),  # Validation 예측
    'arousal': np.array([...]),
    'true_valence': np.array([...]),  # True labels
    'true_arousal': np.array([...])
}
```

**생성 방법**:
```python
# train_ensemble_subtask2a.py의 validation loop에 추가:

if val_ccc > best_val_ccc:
    # 기존 모델 저장
    torch.save(model.state_dict(), ...)

    # ⭐ Validation 예측 저장 (새로 추가)
    val_predictions = {
        'valence': all_val_preds_v.cpu().numpy(),
        'arousal': all_val_preds_a.cpu().numpy(),
        'true_valence': all_val_labels_v.cpu().numpy(),
        'true_arousal': all_val_labels_a.cpu().numpy()
    }
    save_path = f'results/subtask2a/validation_predictions/val_preds_seed{RANDOM_SEED}.npy'
    np.save(save_path, val_predictions)
```

---

## 📈 성능 진행 상황

### Timeline

| 날짜 | 앙상블 | CCC | 변경사항 |
|------|--------|-----|----------|
| 2025-11-20 | 3-model (42+123+777) | 0.6021 | 초기 앙상블 |
| 2025-12-19 | 2-model (123+777) | 0.6305 | seed42 제거 (+6%) |
| 2025-12-21 | 3-model (123+777+888) | 0.6505 (예상) | seed888 추가 |
| 2025-12-22 | 4-model + Specialist | 0.7005 (목표) | Arousal Specialist |
| 2025-12-22 | Stacking | 0.7105 (목표) | Ridge 최적화 |

### 목표별 상태

| 목표 CCC | 상태 | 달성 시점 |
|---------|------|----------|
| 0.60 | ✅ 달성 | 2025-11-20 |
| 0.62 | ✅ 달성 | 2025-12-19 |
| 0.65 | 🎯 진행중 | 2025-12-21 (예상) |
| 0.70 | 🎯 목표 | 2025-12-22 (예상) |
| 0.75 | ⚠️ 도전적 | 미정 |

---

## 🔍 모델 분석

### Valence vs Arousal 성능 갭

| 모델 | Valence CCC | Arousal CCC | 갭 |
|------|-------------|-------------|-----|
| seed777 | 0.7593 | 0.5516 | **-0.21** |
| seed123 | 0.6298 | 0.4362 | -0.19 |
| seed42 | 0.6532 | 0.3574 | **-0.30** |

**문제**: Arousal이 Valence보다 평균 **27% 낮음**

**해결책**: Arousal Specialist 모델
- 목표 Arousal CCC: 0.60-0.65
- 예상 전체 개선: +0.04-0.06

---

### Seed 패턴 분석

| Seed 유형 | Seeds | 최고 CCC | 평균 CCC |
|----------|-------|---------|---------|
| 단순 | 42 | 0.5053 | 0.5053 |
| 연속 | 123 | 0.5330 | 0.5330 |
| **반복** | **777** | **0.6554** | **0.6554** |
| 반복 (예상) | 888, 999 | 0.60-0.63? | 0.61? |

**가설**: 반복 숫자 seed가 우수한 초기화 제공
**검증**: seed888, 999 훈련 후 확인

---

## 💡 사용 가이드

### 1. 새 모델 훈련 후
```bash
# 1. CCC 확인 (훈련 완료 시 출력됨)
# 2. optimal_ensemble.json 업데이트
# 3. model_training_log.json에 기록
```

### 2. 최적 앙상블 찾기
```bash
python scripts/data_analysis/subtask2a/calculate_optimal_ensemble_weights.py

# 출력: optimal_ensemble.json (업데이트됨)
```

### 3. Stacking 최적화 (고급)
```bash
# 사전 조건: validation_predictions/ 디렉토리에 모든 모델의 예측 필요

python scripts/data_analysis/subtask2a/optimize_ensemble_stacking.py

# 출력: stacking_optimization.json
```

### 4. 최종 예측
```bash
# 예측 스크립트에 최신 가중치 적용
# stacking_optimization.json 또는 optimal_ensemble.json 사용

python scripts/data_analysis/subtask2a/predict_test_subtask2a_optimized.py
```

---

## 📝 체크리스트

### 훈련 단계
- [x] seed42 제거 검증 (test_2model_ensemble.py)
- [x] 2-model baseline 확정 (CCC 0.6305)
- [ ] seed888 훈련
- [ ] seed999 훈련 (조건부)
- [ ] Arousal Specialist 훈련
- [ ] 모든 결과 model_training_log.json에 기록

### 최적화 단계
- [ ] Validation 예측 저장 (모든 모델)
- [ ] Stacking 최적화 실행
- [ ] 최종 앙상블 가중치 결정
- [ ] 예측 스크립트 업데이트

### 제출 단계
- [ ] 평가파일 릴리스 대기
- [ ] 최종 예측 생성
- [ ] 예측 검증
- [ ] Codabench 제출
- [ ] 결과 test_results_template.json에 기록

---

**마지막 업데이트**: 2025-12-19
**현재 Baseline**: CCC 0.6305
**목표**: CCC 0.70-0.72
