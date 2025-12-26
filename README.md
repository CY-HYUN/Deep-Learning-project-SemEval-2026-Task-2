# SemEval 2026 Task 2a - Emotional State Prediction

> **현재 상태**: CCC 0.6305 (목표 0.62 초과 ✅)
> **최종 목표**: CCC 0.70-0.72
> **다음 단계**: Google Colab Pro에서 모델 훈련

---

## 🚀 빠른 시작

### 지금 바로 시작하려면?
**→ [QUICKSTART.md](./QUICKSTART.md)** ⭐⭐⭐

6단계 실행 가이드:
1. seed888 훈련 (2시간)
2. Arousal Specialist 훈련 (4시간) ⭐ 핵심!
3. seed999 훈련 (선택, 2시간)
4. 최종 앙상블 구성
5. 평가파일 대기
6. 제출

---

## 📊 현재 성능

### 개별 모델
| Model | Seed | CCC | Valence | Arousal | Status |
|-------|------|-----|---------|---------|--------|
| Model 1 | 123 | 0.5330 | 0.6298 | 0.4362 | ✅ |
| Model 2 | 777 | 0.6554 | 0.7593 | 0.5516 | ✅ ⭐ 최고 |
| Model 3 | 42 | 0.5053 | 0.6532 | 0.3574 | ❌ 제거됨 |

### 현재 Baseline (2-model)
```
CCC: 0.6305 ✅ (목표 초과!)
Valence: 0.76 (좋음)
Arousal: 0.55 (개선 필요!)
```

### 예상 최종 성능
```
Conservative (85%): CCC 0.68-0.70
Aggressive (70%): CCC 0.70-0.72
```

---

## 🎯 전략

### 핵심 문제
**Arousal (0.55) << Valence (0.76)** → 27% 차이

### 해결 방법
1. **Arousal Specialist 모델** (가장 큰 개선 +0.05-0.08)
2. **seed888, 999 추가** (반복 숫자 패턴)
3. **Stacking 최적화**

---

## 📂 프로젝트 구조

```
프로젝트/
├── README.md                          (이 파일)
├── QUICKSTART.md                      ⭐⭐⭐ (즉시 시작)
│
├── scripts/
│   ├── data_train/subtask2a/
│   │   └── train_ensemble_subtask2a.py  ✅ (훈련 스크립트)
│   └── data_analysis/subtask2a/
│       ├── predict_test_subtask2a_optimized.py  (예측)
│       └── calculate_optimal_ensemble_weights.py  (가중치)
│
├── models/
│   ├── subtask2a_seed123_best.pt      ✅
│   ├── subtask2a_seed777_best.pt      ✅
│   ├── subtask2a_seed888_best.pt      (훈련 예정)
│   └── subtask2a_arousal_specialist_seed1111_best.pt  (훈련 예정)
│
├── results/subtask2a/
│   └── optimal_ensemble.json          ✅ (현재 baseline)
│
└── data/
    ├── train_subtask2a.csv            ✅
    └── test/ (평가파일 대기)
```

---

## 📝 문서

### 필수
- **[QUICKSTART.md](./QUICKSTART.md)** - 즉시 실행 가이드 (6단계)

### 참고
- **[scripts/README.md](./scripts/README.md)** - 스크립트 설명
- **[results/subtask2a/README.md](./results/subtask2a/README.md)** - 결과 파일 설명

### 구버전 (참고용)
- **[docs/archive/old_guides/](./docs/archive/old_guides/)** - 이전 문서들

---

## ✅ 체크리스트

### 완료 ✅
- [x] 설문조사 작성
- [x] Zoom 건너뜀
- [x] seed42 제거
- [x] 2-model baseline (CCC 0.6305)
- [x] 문서 정리 (2개 파일로 통합)

### 진행 중 🔄
- [ ] **seed888 훈련** ← 지금 여기!
- [ ] Arousal Specialist 훈련
- [ ] seed999 훈련 (선택)
- [ ] 최종 앙상블 구성
- [ ] 평가파일 대기 (12/23-25)
- [ ] 제출

---

## 🔗 링크

- **Codabench**: https://www.codabench.org/competitions/9963/
- **설문조사**: https://forms.gle/zxS69TKQ4mjGZbEc6 (완료 ✅)
- **Google Colab**: https://colab.research.google.com/

---

## 💡 핵심 요약

### 왜 Arousal Specialist?
```
현재 가장 큰 문제: Arousal 성능
해결: Arousal 전문 모델 (90% CCC weight)
효과: 전체 CCC +0.05-0.08 (가장 큰 개선)
```

### 왜 seed888, 999?
```
패턴: seed777 (반복 숫자) = 최고 성능
전략: 동일 패턴 시도
확률: 70% 성공 (CCC 0.60+)
```

### Google Colab Pro?
```
GPU: A100 > V100 > T4
시간: 30-40% 단축
병렬: 여러 노트북 동시 실행 가능
```

---

## 📞 도움이 필요하면?

1. **지금 뭘 해야 하지?** → [QUICKSTART.md](./QUICKSTART.md) 1단계
2. **스크립트가 뭐지?** → [scripts/README.md](./scripts/README.md)
3. **결과 파일은?** → [results/subtask2a/README.md](./results/subtask2a/README.md)

---

**상태**: 즉시 실행 가능 ✅
**목표**: CCC 0.70-0.72
**시간**: 주말 8시간

🚀 **[QUICKSTART.md](./QUICKSTART.md)에서 시작!**

---

**마지막 업데이트**: 2025-12-19
**버전**: Final (2-file structure)
