# 📚 프로젝트 문서

> **즉시 시작**: 프로젝트 루트의 [QUICKSTART.md](../QUICKSTART.md)를 참고하세요!

---

## 📂 문서 구조

```
docs/
├── README.md                   (이 파일 - 네비게이션)
├── PROJECT_STATUS.md           ⭐ 현재 상태 및 전체 개요
├── NEXT_ACTIONS.md             ⭐ 다음에 할 일
├── TRAINING_STRATEGY.md        ⭐ 성능 향상 전략
└── archive/                    (상세 참고 문서)
    ├── 01_PROJECT_OVERVIEW.md
    ├── 03_SUBMISSION_GUIDE.md
    └── EVALUATION_METRICS_EXPLAINED.md
```

---

## 🚀 시작 가이드

### 1. 지금 뭘 해야 하나요?
**→ [NEXT_ACTIONS.md](NEXT_ACTIONS.md)** ⭐⭐⭐

다음에 할 일이 우선순위별로 정리되어 있습니다:
- 필수: 평가파일 대기
- 선택 A: seed888 훈련 (추천)
- 선택 B: Arousal Specialist 훈련

### 2. 프로젝트 현재 상태가 궁금해요
**→ [PROJECT_STATUS.md](PROJECT_STATUS.md)** ⭐⭐⭐

현재 성능, 파일 상태, 완료/진행 중 작업을 한눈에 볼 수 있습니다.

### 3. 성능을 더 높이고 싶어요
**→ [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md)** ⭐⭐⭐

4단계 성능 향상 전략이 상세히 설명되어 있습니다.

### 4. 즉시 실행하고 싶어요
**→ [../QUICKSTART.md](../QUICKSTART.md)** ⭐⭐⭐

6단계로 나뉜 실행 가이드가 있습니다.

---

## 📊 현재 상태 (2025-12-19)

### 성능
```
Baseline: CCC 0.6305 ✅ (목표 0.62 초과)
Valence: 0.76 (좋음)
Arousal: 0.55 (개선 필요)
```

### 완료 ✅
- 설문조사 작성
- Zoom 미팅 (건너뜀)
- seed42 제거 (성능 향상)
- 2-model baseline 확정
- 문서 정리 및 복구

### 진행 중 🔄
- seed888 훈련 준비 (선택)

### 대기 중 ⏳
- 평가파일 릴리스 (12/23-25 예상)

---

## 📖 핵심 문서

### [NEXT_ACTIONS.md](NEXT_ACTIONS.md) ⭐ 가장 중요!
**내용**: 다음에 할 일 (우선순위별)
- 필수 작업 (평가파일 대기 및 제출)
- 선택 작업 A (seed888 훈련, 2시간)
- 선택 작업 B (Arousal Specialist, 4시간)
- 상세한 실행 방법
- 타임라인 및 결정 트리

**언제 읽나요?**: 지금 바로! (가장 먼저)

---

### [PROJECT_STATUS.md](PROJECT_STATUS.md)
**내용**: 프로젝트 전체 상황 파악
- 현재 성능 및 모델 상태
- 완료/진행/대기 작업
- 파일 상태 (모델, 스크립트, 결과)
- 타임라인 및 시나리오
- 기술 스택

**언제 읽나요?**: 전체 상황이 궁금할 때

---

### [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md)
**내용**: 성능 향상 4단계 전략
- 1단계: seed888 (2시간, +0.015)
- 2단계: Arousal Specialist (4시간, +0.050)
- 3단계: Stacking 최적화 (2시간, +0.010)
- 4단계: seed999 (2시간, +0.010)
- 각 단계별 상세 방법 및 코드
- 예상 결과 및 시나리오

**언제 읽나요?**: 성능을 높이고 싶을 때

---

## 🗂️ 참고 문서 (archive/)

### [01_PROJECT_OVERVIEW.md](archive/01_PROJECT_OVERVIEW.md)
**내용**: 프로젝트 전체 배경
- SemEval 2026 Task 2 공식 요구사항
- 교수님 평가 기준
- 논문 작성 가이드
- 팀 프로젝트 규칙

**언제 읽나요?**: 평가 기준이 궁금할 때, 논문 작성 시

---

### [03_SUBMISSION_GUIDE.md](archive/03_SUBMISSION_GUIDE.md)
**내용**: Codabench 제출 상세 가이드
- 평가파일 다운로드
- 예측 생성 방법
- 제출 파일 검증
- Troubleshooting

**언제 읽나요?**: 평가파일 릴리스 후, 제출 준비 시

---

### [EVALUATION_METRICS_EXPLAINED.md](archive/EVALUATION_METRICS_EXPLAINED.md)
**내용**: 평가 지표 상세 설명
- Pearson r vs CCC
- Subtask 1 vs 2a 차이
- Between-user / Within-user correlation

**언제 읽나요?**: 평가 지표가 궁금할 때, 논문 작성 시

---

## 🎯 빠른 네비게이션

### 상황별 문서 찾기

| 상황 | 문서 |
|------|------|
| 지금 뭘 해야 하지? | [NEXT_ACTIONS.md](NEXT_ACTIONS.md) |
| 현재 상태가 어떻게 되지? | [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| 성능을 높이고 싶어 | [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md) |
| 바로 실행하고 싶어 | [../QUICKSTART.md](../QUICKSTART.md) |
| 제출은 어떻게 하지? | [archive/03_SUBMISSION_GUIDE.md](archive/03_SUBMISSION_GUIDE.md) |
| 평가 기준이 뭐지? | [archive/01_PROJECT_OVERVIEW.md](archive/01_PROJECT_OVERVIEW.md) |
| 평가 지표가 뭐지? | [archive/EVALUATION_METRICS_EXPLAINED.md](archive/EVALUATION_METRICS_EXPLAINED.md) |

---

## 📁 프로젝트 루트 문서

### [README.md](../README.md)
- 프로젝트 개요
- 현재 성능 요약
- 전략 개요
- 프로젝트 구조

### [QUICKSTART.md](../QUICKSTART.md) ⭐
- 6단계 실행 가이드
- 각 단계별 명령어
- Troubleshooting
- 타임라인

---

## 🔗 외부 링크

### 공식
- **Codabench**: https://www.codabench.org/competitions/9963/
- **설문조사**: https://forms.gle/zxS69TKQ4mjGZbEc6 (완료 ✅)
- **Google Colab**: https://colab.research.google.com/

### 내부
- **스크립트 설명**: [../scripts/README.md](../scripts/README.md)
- **결과 파일**: [../results/subtask2a/README.md](../results/subtask2a/README.md)

---

## 💡 추천 읽기 순서

### 처음 시작하는 경우
1. [NEXT_ACTIONS.md](NEXT_ACTIONS.md) - 다음 할 일
2. [PROJECT_STATUS.md](PROJECT_STATUS.md) - 현재 상태
3. [../QUICKSTART.md](../QUICKSTART.md) - 실행 방법

### seed888 훈련을 고려하는 경우
1. [NEXT_ACTIONS.md](NEXT_ACTIONS.md) - 선택 작업 A
2. [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md) - 1단계 상세
3. [../QUICKSTART.md](../QUICKSTART.md) - 1단계 실행

### 평가파일 릴리스 후
1. [NEXT_ACTIONS.md](NEXT_ACTIONS.md) - 필수 작업
2. [archive/03_SUBMISSION_GUIDE.md](archive/03_SUBMISSION_GUIDE.md) - 제출 가이드
3. [../QUICKSTART.md](../QUICKSTART.md) - 6단계 실행

### 논문/보고서 작성 시
1. [PROJECT_STATUS.md](PROJECT_STATUS.md) - 전체 정리
2. [archive/01_PROJECT_OVERVIEW.md](archive/01_PROJECT_OVERVIEW.md) - 배경
3. [archive/EVALUATION_METRICS_EXPLAINED.md](archive/EVALUATION_METRICS_EXPLAINED.md) - 지표

---

## 📋 문서 정리 내역 (2025-12-19)

### 새로 만든 핵심 문서 ✅
- **PROJECT_STATUS.md** - 프로젝트 전체 상태 및 개요
- **NEXT_ACTIONS.md** - 다음 할 일 상세 가이드
- **TRAINING_STRATEGY.md** - 성능 향상 4단계 전략
- **README.md** (업데이트) - 문서 네비게이션

### 유지된 참고 문서 ✅
- **01_PROJECT_OVERVIEW.md** - 프로젝트 배경 및 평가 기준
- **03_SUBMISSION_GUIDE.md** - 상세 제출 가이드
- **EVALUATION_METRICS_EXPLAINED.md** - 평가 지표 설명

---

## 📊 문서 개요

| 문서 | 크기 | 용도 | 중요도 |
|------|------|------|--------|
| NEXT_ACTIONS.md | 중 | 다음 할 일 | ⭐⭐⭐ 최고 |
| PROJECT_STATUS.md | 대 | 전체 상태 | ⭐⭐⭐ |
| TRAINING_STRATEGY.md | 대 | 성능 향상 | ⭐⭐⭐ |
| 01_PROJECT_OVERVIEW.md | 대 | 배경 | ⭐⭐ 참고용 |
| 03_SUBMISSION_GUIDE.md | 중 | 제출 | ⭐⭐ 제출 시 |
| EVALUATION_METRICS_EXPLAINED.md | 중 | 평가 | ⭐ 논문 시 |

---

**최신 상태**: ✅ 2025-12-19
**다음 읽을 문서**: [NEXT_ACTIONS.md](NEXT_ACTIONS.md) ⭐
**즉시 실행**: [../QUICKSTART.md](../QUICKSTART.md)
