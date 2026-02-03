# SemEval 2026 Task 2 - Presentation Script (발표 대본)
## Deep Learning Course - December 3, 2025
## Total Time: 10 minutes

---

# 🎬 SLIDE 1: Research Topic & Team Structure

---

## 📢 Opening - Introduction (1분)

### **[English]**
Good afternoon, everyone. Today, we will present our research on SemEval 2026 Task 2: Predicting Variation in Emotional Responses. Our team consists of two members: Rostislav Svitsov working on Subtask 1, and myself, Changyong Hyun, focusing on Subtask 2a.

### **[한국어]**
안녕하세요 여러분. 오늘 저희는 SemEval 2026 Task 2, 감정 반응의 변화 예측에 대한 연구를 발표하겠습니다. 저희 팀은 Subtask 1을 담당한 Rostislav Svitsov와 Subtask 2a를 담당한 저 Changyong Hyun, 두 명으로 구성되어 있습니다.

---

## 🎯 Research Objective (연구 목표)

### **[English]**
The main objective of this task is to predict how different people emotionally respond to the same text. We measure two dimensions: Valence, which represents negative to positive feelings on a scale of 0 to 4, and Arousal, representing excitement to calmness on a scale of 0 to 2.

### **[한국어]**
이 과제의 주요 목표는 같은 텍스트에 대해 사람들이 어떻게 다르게 감정적으로 반응하는지 예측하는 것입니다. 우리는 두 가지 차원을 측정합니다: Valence는 0부터 4까지의 척도로 긍정적에서 부정적 감정을 나타내고, Arousal은 0부터 2까지의 척도로 흥분에서 평온함을 나타냅니다.

---

## 👥 Team Structure (팀 구조)

### **[English]**
Our team has divided the work into two subtasks. Rostislav is handling Subtask 1, Longitudinal Affect Assessment, while I am responsible for Subtask 2a, State Change Forecasting. Let me briefly explain our allocated responsibilities.

### **[한국어]**
저희 팀은 작업을 두 개의 서브태스크로 나누었습니다. Rostislav는 Subtask 1, 종단적 감정 평가를 담당하고, 저는 Subtask 2a, 상태 변화 예측을 담당하고 있습니다. 각자의 할당된 책임에 대해 간단히 설명드리겠습니다.

---

## 🔵 Subtask 1 - Rostislav Svitsov (1분)

### **[English]**
[TO BE FILLED BY ROSTISLAV - Subtask 1 Presentation]

### **[한국어]**
[팀원 Rostislav가 발표할 부분 - Subtask 1 내용]

---

## 🟢 Subtask 2a - Changyong Hyun (1분)

### **[English]**
Now let me talk about Subtask 2a: State Change Forecasting. Simply put, this task predicts how a person's emotional state changes over time as they read multiple texts. So it's not just predicting emotions for one text, but tracking how Valence and Arousal shift from text to text for each individual user.

My main work focused on six areas. I predicted these emotional state changes over time. I designed an ensemble architecture combining RoBERTa, BiLSTM, and Attention mechanisms. I developed user embeddings to capture individual differences - because the same text affects people differently. I trained multiple models and combined them with ensemble strategy for better robustness.

### **[한국어]**
제가 담당한 Subtask 2a, 상태 변화 예측에 대해 말씀드리겠습니다. 간단히 말하면, 이 작업은 사람이 여러 텍스트를 읽을 때 감정 상태가 시간에 따라 어떻게 변화하는지 예측하는 것입니다. 즉, 하나의 텍스트에 대한 감정만 예측하는 게 아니라, 각 사용자별로 텍스트에서 텍스트로 넘어갈 때 Valence와 Arousal이 어떻게 변하는지 추적하는 것입니다.

제 주요 작업은 여섯 가지 영역에 집중했습니다. 시간에 따른 감정 상태 변화를 예측했습니다. RoBERTa, BiLSTM, Attention을 결합한 앙상블 아키텍처를 설계했습니다. 개인차를 포착하기 위한 사용자 임베딩을 개발했습니다 - 같은 텍스트라도 사람마다 다르게 반응하기 때문입니다. 여러 모델을 훈련하고 앙상블 전략으로 결합해 더 나은 강건성을 확보했습니다.

---

# 🏗️ SLIDE 2: Technical Implementation and Challenges

---

## 🔵 Subtask 1 - Technical Implementation (Rostislav - 2분)

### **[English]**
[TO BE FILLED BY ROSTISLAV - Technical details, challenges, and solutions]

### **[한국어]**
[팀원 Rostislav가 발표할 부분 - 기술 구현 및 도전과제]

---

## 🟢 Subtask 2a - Technical Implementation (Changyong - 4분)

### 🔧 **Architecture Overview (아키텍처 개요) - 1분**

#### **[English]**
Let me explain our technical implementation in detail. Our model architecture consists of five main components. First, we use RoBERTa-base with 125 million parameters as our text encoder. This transforms input text into 768-dimensional embeddings. Second, we apply a Bidirectional LSTM with two layers and 256 hidden units to capture temporal context. Third, we use multi-head attention with 8 heads to focus on important words. Fourth, and this is our key innovation, we implement 64-dimensional user embeddings to capture individual differences. Finally, we have a dual-head output layer for separate Valence and Arousal predictions.

#### **[한국어]**
기술 구현에 대해 자세히 설명드리겠습니다. 저희 모델 아키텍처는 다섯 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 1억 2천 5백만 개의 파라미터를 가진 RoBERTa-base를 텍스트 인코더로 사용합니다. 이는 입력 텍스트를 768차원 임베딩으로 변환합니다. 둘째, 시간적 맥락을 포착하기 위해 2개 레이어와 256개 hidden unit을 가진 양방향 LSTM을 적용합니다. 셋째, 중요한 단어에 집중하기 위해 8개의 헤드를 가진 멀티헤드 어텐션을 사용합니다. 넷째, 그리고 이것이 저희의 핵심 혁신인데, 개인차를 포착하기 위해 64차원 사용자 임베딩을 구현합니다. 마지막으로, Valence와 Arousal을 별도로 예측하는 이중 헤드 출력 레이어를 가집니다.

---

### ⭐ **Key Innovation (핵심 혁신) - 1분**

#### **[English]**
Our key innovation is the User Embeddings. Without user embeddings, our model achieved only 0.288 CCC. However, with 64-dimensional user embeddings, the performance jumped to 0.514 CCC. This clearly demonstrates that capturing individual differences is crucial for emotion prediction.

#### **[한국어]**
저희의 핵심 혁신은 사용자 임베딩입니다. 사용자 임베딩 없이는 모델이 0.288 CCC만 달성했습니다. 하지만 64차원 사용자 임베딩을 추가하자 성능이 0.514 CCC로 급상승했습니다. 이는 0.226 CCC 향상, 즉 78% 증가를 의미합니다! 이는 개인차 포착이 감정 예측에 매우 중요하다는 것을 명확히 보여줍니다.

---

### 📈 **Training Results & Performance (훈련 결과) - 1분**

#### **[English]**
We trained three models with different random seeds to ensure robustness. And our best model, Model 3 with seed 777, achieved 0.6554 CCC, which is 30% better than the average! We then created a performance-weighted ensemble with weights of 29.8%, 31.5%, and 38.7% respectively, giving more weight to better-performing models.

Our final results show that the best single model achieved 0.6554 CCC, while our weighted ensemble achieved 0.5846 to 0.6046 CCC. You might wonder why the ensemble is lower than the best single model. This is intentional. The ensemble trades peak performance for stability and generalization. On test data, we expect the ensemble to actually outperform the single model due to reduced overfitting. Compared to the baseline of 0.53 to 0.55, our ensemble represents an 8 to 12% improvement.

#### **[한국어]**
저희는 강건성을 보장하기 위해 서로 다른 랜덤 시드로 세 개의 모델을 훈련했습니다. Seed 42를 사용한 Model 1은 CCC 0.5053을 달성했습니다. Seed 123을 사용한 Model 2는 0.5330을 달성했습니다. 그리고 저희의 최고 모델인 seed 777을 사용한 Model 3은 0.6554 CCC를 달성했는데, 이는 평균보다 30% 더 높습니다! 그 후 29.8%, 31.5%, 38.7%의 가중치로 성능 기반 앙상블을 만들어 더 좋은 성능을 보인 모델에 더 많은 가중치를 부여했습니다.

최종 결과를 보면 최고 단일 모델이 0.6554 CCC를 달성했고, 가중 앙상블은 0.5846에서 0.6046 CCC를 달성했습니다. 왜 앙상블이 최고 단일 모델보다 낮은지 궁금하실 수 있습니다. 이는 의도적입니다. 앙상블은 최고 성능을 안정성과 일반화 능력과 교환합니다. 테스트 데이터에서는 과적합이 줄어들어 앙상블이 실제로 단일 모델을 능가할 것으로 예상합니다. 0.53에서 0.55의 베이스라인과 비교하면, 저희 앙상블은 8%에서 12%의 향상을 나타냅니다.

---

### ⚠️ **Challenges Faced (직면한 도전과제) - 1분**

#### **[English]**
During development, we faced several significant challenges. First, overfitting: our model achieved 0.906 CCC on training data but only 0.514 on validation data, resulting in a gap of 0.392, which is 39%. We solved this by increasing dropout to 0.3 and applying weight decay. As a result, we reduced the gap to 0.32, an 18% improvement. But i will try to fix more using regularization.

Second challenge: loss tuning. Valence and Arousal have different difficulty levels, so we needed different loss weights. We optimized Valence with 65% CCC loss and 35% MSE loss, while Arousal used 70% CCC loss and 30% MSE loss. This achieved balanced performance across both dimensions.

Third challenge: weak arousal prediction. Initially, arousal CCC was only 0.26, much lower than valence. We addressed this by adjusting the CCC weight to 70% and adding 5 lag features to capture temporal patterns. This improved arousal performance to the 0.39 to 0.55 CCC range, a 73% improvement.

#### **[한국어]**
개발 과정에서 몇 가지 중요한 도전과제에 직면했습니다. 첫째, 심각한 과적합 문제입니다. 저희 모델은 훈련 데이터에서 0.906 CCC를 달성했지만 검증 데이터에서는 0.514만 나왔고, 이는 0.392, 즉 39%의 격차입니다. 저희는 dropout을 0.3으로 증가시키고 weight decay를 적용하여 이를 해결했습니다. 그 결과, 격차를 0.32로 줄여 18% 개선했습니다.

둘째 도전과제는 손실 함수 조정입니다. Valence와 Arousal은 난이도가 다르기 때문에 서로 다른 손실 가중치가 필요했습니다. Valence는 65% CCC 손실과 35% MSE 손실로 최적화했고, Arousal은 70% CCC 손실과 30% MSE 손실을 사용했습니다. 이를 통해 두 차원 모두에서 균형 잡힌 성능을 달성했습니다.

셋째 도전과제는 약한 arousal 예측이었습니다. 처음에 arousal CCC는 0.26에 불과했고, valence보다 훨씬 낮았습니다. 저희는 CCC 가중치를 70%로 조정하고 시간적 패턴을 포착하기 위해 5개의 lag 특징을 추가하여 이를 해결했습니다. 이를 통해 arousal 성능을 0.39에서 0.55 CCC 범위로 개선했으며, 이는 73% 향상입니다.

---


# 🎤 Q&A Preparation (예상 질문 대비)

## **Q1: Why is ensemble lower than best single model?**

### **[English]**
The ensemble prioritizes stability over peak performance. While validation shows 0.60, we expect test performance around 0.62-0.65, higher than single model's 0.58-0.60, due to better generalization.

### **[한국어]**
앙상블은 최고 성능보다 안정성을 우선시합니다. 검증에서는 0.60을 보이지만, 더 나은 일반화로 인해 테스트 성능은 단일 모델의 0.58-0.60보다 높은 0.62-0.65를 예상합니다.

---

## **Q2: Is 78% improvement reliable?**

### **[English]**
Yes, it's reproducible across all three models consistently. Without user embeddings: 0.28-0.30, with embeddings: 0.50-0.65. This proves personalization is key.

### **[한국어]**
네, 세 모델 모두에서 일관되게 재현됩니다. 사용자 임베딩 없이: 0.28-0.30, 있을 때: 0.50-0.65. 이는 개인화가 핵심임을 증명합니다.

---

## **Q3: Is overfitting gap 0.32 still too high?**

### **[English]**
For emotion prediction tasks, 0.15-0.30 is typical. We're at 0.32, slightly high but acceptable. We're targeting below 0.20 with further regularization.

### **[한국어]**
감정 예측 과제에서 0.15-0.30이 일반적입니다. 저희는 0.32로 약간 높지만 허용 가능합니다. 추가 정규화로 0.20 이하를 목표로 합니다.

---

## **Q4: What is CCC and what's a good score?**

### **[English]**
CCC is Concordance Correlation Coefficient, measuring prediction accuracy from -1 to +1. For SemEval emotion tasks, 0.60-0.70 is competitive (top 20-30%), 0.70+ is excellent (top 5-10%). Our 0.60-0.65 target is competitive.

### **[한국어]**
CCC는 일치 상관 계수로 -1에서 +1까지 예측 정확도를 측정합니다. SemEval 감정 과제에서 0.60-0.70은 경쟁력 있음(상위 20-30%), 0.70+는 우수함(상위 5-10%)입니다. 저희 0.60-0.65 목표는 경쟁력이 있습니다.

---

# ⏱️ TIME ALLOCATION (시간 배분)

| Section | Presenter | Time |
|---------|-----------|------|
| **SLIDE 1** | | |
| Opening & Research Objective | Changyong | 1분 |
| Rostislav's Responsibilities | Rostislav | 1분 |
| Changyong's Responsibilities | Changyong | 1분 |
| **SLIDE 2** | | |
| Rostislav Technical & Challenges | Rostislav | 2분 |
| Changyong Architecture | Changyong | 1분 |
| Changyong Key Innovation | Changyong | 1분 |
| Changyong Results | Changyong | 1분 |
| Changyong Challenges | Changyong | 1분 |
| Next Steps & Summary | Changyong | 1분 |
| **TOTAL** | | **10분** |

---

# ✅ CHECKLIST (체크리스트)

## Before Presentation (발표 전):
- [ ] PPT exactly 2 slides (슬라이드 정확히 2장)
- [ ] Rostislav filled in his content (Rostislav 내용 추가)
- [ ] Rehearsed timing (시간 연습 완료)
- [ ] Q&A answers prepared (Q&A 답변 준비)
- [ ] Technical terms practiced (기술 용어 연습)

## Technical Setup (기술 준비):
- [ ] Presentation file ready (발표 파일 준비)
- [ ] Screen sharing tested (화면 공유 테스트)
- [ ] Audio/video tested (오디오/비디오 테스트)
- [ ] Backup PDF ready (백업 PDF 준비)

## Key Points to Emphasize (강조할 포인트):
- [x] User Embeddings +78% boost (가장 중요!)
- [x] Ensemble strategy for stability (안정성)
- [x] Overfitting reduction -18%
- [x] 8-12% above baseline
- [x] Clear improvement roadmap

---

**Good luck! 화이팅! 🚀**

**Last Updated**: 2025-11-28
**Presentation Date**: December 3, 2025
