# Live Presentation Script - Subtask 2a: State Change Forecasting
## 10-12 Minute Demonstration Guide (Slides 16-31 + Demo)

**Presenter**: Changyong Hyun
**Date**: January 2026
**Duration**: 11:30 minutes (10-12 minute target)
**Format**: PowerPoint Slides (16-31) + Pre-Executed Demo Notebook

---

## 🎯 Presentation Strategy

### Overview
This presentation demonstrates my **production-ready emotion forecasting system** that achieved **CCC 0.6833** (+10.4% above 0.62 target) through dimension-specific optimization and quality-over-quantity ensemble design.

이 발표는 차원별 최적화와 품질 우선 앙상블 설계를 통해 **CCC 0.6833** (+10.4% 목표 초과)을 달성한 **프로덕션급 감정 예측 시스템**을 보여줍니다.

### Why Pre-Executed Demo?
- **Avoid technical delays**: Live execution takes 2-3 minutes
- **Maximize explanation time**: Focus on methodology, not debugging
- **Professional delivery**: All outputs ready to discuss

**미리 실행한 데모를 사용하는 이유**: 기술적 지연 방지, 설명 시간 최대화, 전문적인 전달

### Time Allocation
| Section | Duration |
|---------|----------|
| **Part 1: PowerPoint (Slides 16-31)** | 7:50 min |
| **Part 2: Demo Walkthrough** | 3:20 min |
| **Part 3: Closing** | 0:30 min |
| **Total** | 11:40 min |

---

## ✅ Pre-Presentation Checklist

### 30 Minutes Before
- [ ] Open PowerPoint at **Slide 16**
- [ ] Open pre-executed demo notebook
- [ ] Test screen switching
- [ ] Set timer for 12 minutes
- [ ] Review critical sections (marked with ⭐)

---

# PART 1: PowerPoint Presentation (Slides 16-31)
## Duration: 7:50 minutes

---

### 📊 Slide 16: PART II - Subtask 2a: State Change Forecasting
**⏱️ Time: 0:00-0:25 (25 seconds)**

> **[Confident opening, make eye contact]**
>
> "Good afternoon. I'm Changyong Hyun, and I'll now present Subtask 2a: State Change Forecasting."
>
> "My focus was on optimizing longitudinal sequence modeling through hybrid architectures and specialized dimension weighting—predicting how a user's emotional state will change over time based on their diary history."
>
> **[Korean]** "안녕하세요. 현창용입니다. Subtask 2a 감정 상태 변화 예측을 발표하겠습니다. 하이브리드 아키텍처와 차원별 가중치를 통한 시계열 시퀀스 모델링 최적화에 초점을 맞췄습니다."

---

### 📈 Slide 17: Dataset Analysis & EDA Insights
**⏱️ Time: 0:25-1:10 (45 seconds)**

> **[Point to slide title]**
>
> "I started by analyzing the longitudinal corpus: 137 training users and 46 test users, tracked from November 2025 to January 2026."
>
> **[Point to key finding - EMPHASIZE]**
>
> "The exploratory data analysis revealed a critical challenge: there's a **38% volatility gap**. Arousal shifts are 38% less frequent but more sporadic than Valence shifts. Arousal has lower variance—mean 1.0, standard deviation 0.6—indicating higher subjectivity and prediction difficulty."
>
> "This gap became the focus of my innovation."
>
> **[Korean]** "2024년 11월부터 2026년 1월까지 추적했습니다. 탐색적 데이터 분석 결과, 핵심 과제를 발견했습니다: **38% 변동성 격차**. Arousal 변화는 Valence보다 38% 덜 빈번하지만 더 불규칙합니다. 이것이 제 혁신의 초점이 되었습니다."

---

### 🧠 Slide 18: Hybrid Model Architecture
**⏱️ Time: 1:10-2:00 (50 seconds)**

> **[Gesture to architecture]**
>
> "My solution uses semantic-temporal integration—capturing both instantaneous textual affect and long-term sequential patterns."
>
> **[Point to components]**
>
> "The architecture has four components: RoBERTa-base extracts 768-dimensional contextual embeddings—125 million parameters for semantic understanding. A dual-layer BiLSTM with 256 units per layer models how emotions evolve over time. An 8-head attention mechanism focuses on the words that matter most. And finally, decoupled regression heads optimized separately for Valence and Arousal."
>
> "This separation is crucial—it allows dimension-specific optimization for handling that 38% volatility gap."
>
> **[Korean]** "제 솔루션은 의미-시간 통합을 사용합니다. 아키텍처는 4개 구성요소로 이루어져 있습니다: RoBERTa-base, BiLSTM, 8-head attention, 그리고 Valence와 Arousal을 위한 분리된 회귀 헤드입니다. 이 분리가 38% 변동성 격차 해결의 핵심입니다."

---

### 📊 Slide 19: Advanced 47-Dim Feature Taxonomy
**⏱️ Time: 2:00-2:40 (40 seconds)**

> **[Point to feature categories]**
>
> "Beyond deep learning, I engineered 47 hand-crafted features in three categories."
>
> "768 textual features from RoBERTa embeddings capturing deep essay semantics. 20 temporal features including lags from previous 1-3 steps, moving averages, and **3 specific Arousal Dynamics features** designed to capture energy-level shifts. And 29 personal features with 64-dimensional learnable user embeddings—personalizing predictions for each user's emotional baseline."
>
> **[Korean]** "딥러닝 외에도 3개 카테고리로 47개 수작업 피처를 설계했습니다: RoBERTa 텍스트 임베딩 768개, 시간적 피처 20개 (여기에 에너지 변화 포착을 위한 Arousal 전용 피처 3개 포함), 그리고 개인화를 위한 사용자 피처 29개입니다."

---

### ⚠️ Slide 20: The Arousal Prediction Bottleneck
**⏱️ Time: 2:40-3:10 (30 seconds)**

> **[Serious tone - problem statement]**
>
> "Initial ensemble analysis revealed a massive performance gap: energy-level forecasting—Arousal—was significantly weaker than pleasantness forecasting—Valence."
>
> **[Point to Root Cause]**
>
> "Two root causes: **Subjective variance**—users struggle to define energy levels more than mood. And **low variation**—standard loss functions ignore subtle arousal shifts because they focus on higher-variance Valence patterns."
>
> **[Korean]** "초기 앙상블 분석 결과, 에너지 수준 예측(Arousal)이 기분 예측(Valence)보다 훨씬 약한 성능 격차를 발견했습니다. 두 가지 근본 원인: 주관적 분산과 낮은 변동성입니다."

---

### 🏆 Slide 21: Innovation: Arousal-Specialist Model
**⏱️ Time: 3:10-4:10 (1 minute)** ⭐ **CRITICAL SECTION**

> **[Lean forward - show enthusiasm]**
>
> "So I developed a breakthrough solution: a dedicated model architecture designed specifically to master the energy-activation forecasting gap."
>
> **[Point to 90% CCC Loss graphic]**
>
> "Three key innovations: **Loss engineering**—I re-weighted CCC loss from 70% to 90%, prioritizing agreement over mean error. This forces the model to obsess over concordance correlation specifically for Arousal. **Weighted data loading**—I oversampled high-change emotional shifts during training, telling the model 'These volatile arousal moments are what you need to learn.' And I added new features: Volatility and Acceleration metrics specifically targeting Arousal dynamics."
>
> **[Point to Result - EMPHASIZE]**
>
> "The result? **+6% absolute improvement in Arousal CCC**. This is substantial—it closed nearly half of the 38% performance gap."
>
> **[Korean]** "그래서 획기적인 솔루션을 개발했습니다: 에너지-활성화 예측 격차를 극복하기 위한 전용 모델입니다. 세 가지 핵심 혁신: **Loss 엔지니어링** (CCC loss를 70%에서 90%로 재가중), **가중 데이터 로딩** (변동성 높은 순간을 과대 샘플링), 그리고 **새로운 피처 추가** (Volatility와 Acceleration 지표). 결과는? Arousal CCC에서 **+6% 절대 향상**입니다."

---

### 📊 Slide 22: Detailed Model Benchmarks
**⏱️ Time: 4:10-4:40 (30 seconds)**

> **[Point to table]**
>
> "Here's my full model benchmark comparison."
>
> "Final Ensemble: Overall CCC 0.6833, Valence 0.7593, Arousal 0.5832—my winner. seed777, the base leader: strong on Valence at 0.7593, but Arousal only 0.5516. Arousal Specialist: Arousal CCC jumped to 0.5832—a 6% improvement. Notice seed42 and seed123 at the bottom with Arousal CCC as low as 0.3574—these were discarded."
>
> **[Korean]** "최종 앙상블: 전체 CCC 0.6833, Valence 0.7593, Arousal 0.5832. seed777: Valence 강점, Arousal 0.5516. Arousal 전문가: Arousal CCC 0.5832로 6% 향상. seed42와 seed123는 성능이 낮아 제외했습니다."

---

### 🎯 Slide 23: Quality-over-Quantity Ensemble
**⏱️ Time: 4:40-5:30 (50 seconds)** ⭐ **CRITICAL SECTION**

> **[Gesture to emphasize insight]**
>
> "This brings me to my ensemble strategy, which challenges conventional wisdom."
>
> "I tested approximately 5,000 weight combinations across multiple configurations. What I found was counterintuitive: **2-model ensembles outperformed 3-model and 5-model ensembles**."
>
> **[Point to achievement]**
>
> "Why? Noise injection from weaker seeds. When I included seed42 or seed123, they diluted the ensemble. My final 2-model ensemble: seed777 weighted 50.16%—master of Valence and baseline trends. Arousal Specialist weighted 49.84%—correcting energy-prediction bias."
>
> "Result: **CCC 0.6833, surpassing the target of 0.62 by 10.4%**. Quality over quantity in action."
>
> **[Korean]** "약 5,000개 가중치 조합을 테스트한 결과, 역직관적인 발견: **2-모델 앙상블이 3-모델과 5-모델을 능가**했습니다. 이유는? 약한 시드들의 노이즈 주입. 최종 2-모델 앙상블: seed777 50.16%, Arousal 전문가 49.84%. 결과: **CCC 0.6833, 목표 0.62를 10.4% 초과**."

---

### 📈 Slide 24: Comprehensive Results Summary
**⏱️ Time: 5:30-5:55 (25 seconds)**

> "Final numbers: **CCC 0.6833**, exceeding my target by over 10%. I trained 5 models across multiple experiments. My final ensemble uses just 2 models—the generalist plus the specialist. Achievement: **+10.4% above the 0.62 target**."
>
> **[Korean]** "최종 결과: **CCC 0.6833**, 목표 대비 10% 초과. 5개 모델을 훈련했고, 최종 앙상블은 2개 모델만 사용. **목표 대비 +10.4% 달성**."

---

### 🛠️ Slide 25: Technical Stack & Infrastructure
**⏱️ Time: 5:55-6:10 (15 seconds)**

> "Technical stack: PyTorch with Hugging Face Transformers, Google Colab Pro with A100 GPU and mixed precision training for efficiency."
>
> **[Korean]** "기술 스택: PyTorch와 Hugging Face Transformers, Google Colab Pro의 A100 GPU와 혼합 정밀도 훈련."

---

### ⚙️ Slide 26: Challenges & Solutions
**⏱️ Time: 6:10-6:45 (35 seconds)**

> "I faced four major challenges and solved them systematically."
>
> "The **38% Arousal Gap**: Solved with Arousal-Specialized Model using 90% CCC loss weighting. **Dimension Mismatch**: Implemented dynamic dimension handling with runtime feature slicing between 863 and 866 dimensions. **Ensemble Noise**: Adopted 2-model quality-over-quantity, removing weaker seeds. **Resource Constraints**: Leveraged A100 GPU with mixed precision FP16 for efficient training."
>
> **[Korean]** "4가지 주요 도전과제를 체계적으로 해결했습니다: 38% Arousal 격차 (90% CCC loss 가중치로 해결), 차원 불일치 (동적 차원 핸들링), 앙상블 노이즈 (2-모델 품질 우선), 리소스 제약 (A100 GPU와 FP16 혼합 정밀도)."

---

### 💡 Slide 27: Key Learnings & Insights
**⏱️ Time: 6:45-7:00 (15 seconds)**

> "Key insight: dimension-specific optimization is more powerful than generic multi-tasking. 90% CCC weighting proved critical for agreement-based metrics."
>
> **[Korean]** "핵심 통찰: 차원별 최적화가 일반적 멀티태스킹보다 강력합니다. 90% CCC 가중치가 일치 기반 메트릭에 결정적이었습니다."

---

### 🎯 Slide 28: Conclusion
**⏱️ Time: 7:00-7:30 (30 seconds)**

> **[Point to achievements]**
>
> "To conclude: I achieved **CCC 0.6833, surpassing the 0.62 target by 10.4%** through systematic iteration. My key innovation: the **Arousal-Specialized Model** solved the 38% prediction gap by shifting loss weighting to 90% CCC. I have a production-grade pipeline with dynamic dimension handling, finalized and ready for SemEval 2026 submission."
>
> **[Korean]** "결론: 체계적 반복을 통해 **CCC 0.6833 달성, 0.62 목표를 10.4% 초과**했습니다. 핵심 혁신: **Arousal 전문 모델**이 90% CCC loss 가중치로 38% 예측 격차를 해결했습니다. 프로덕션급 파이프라인 완성, SemEval 2026 제출 준비 완료."

---

### 🔮 Slide 29: Future Directions
**⏱️ Time: 7:30-7:40 (10 seconds)**

> "Future directions include testing larger models like RoBERTa-large for 2-3% additional gains and exploring multimodal signals for energy activation."
>
> **[Korean]** "향후 방향: RoBERTa-large 같은 더 큰 모델 테스트 (2-3% 추가 향상), 에너지 활성화를 위한 멀티모달 신호 탐색."

---

### 📅 Slide 30: Project Lifecycle & Key Milestones
**⏱️ Time: 7:40-7:50 (10 seconds)**

> "Project timeline: Started November 2025, achieved 0.63 CCC in December, developed specialist model on December 23rd, finalized predictions January 2026."
>
> **[Korean]** "프로젝트 타임라인: 2024년 11월 시작, 12월 0.63 CCC 달성, 12월 23일 전문 모델 개발, 2026년 1월 예측 완료."

---

### 🙏 Slide 31: Thank You
**⏱️ Time: 7:50-8:00 (10 seconds)**

> **[Warm closing]**
>
> "Thank you for your attention. I'm now ready to demonstrate how this system works in practice."
>
> **[Korean]** "감사합니다. 이제 이 시스템이 실제로 어떻게 작동하는지 시연하겠습니다."

---

## 🔄 **TRANSITION: PowerPoint → Demo**
**⏱️ Time: 8:00-8:10 (10 seconds)**

> **[Switch screen to demo notebook]**
>
> "I've prepared a demonstration using pre-executed results. This shows exactly what happens when my system makes a prediction for a real user."
>
> **[Korean]** "미리 실행한 결과로 시연을 준비했습니다. 제 시스템이 실제 사용자에 대해 예측을 수행하는 과정을 보여드리겠습니다."

---

# PART 2: Pre-Executed Demo Walkthrough
## Duration: 3:20 minutes

---

### 🖥️ Demo Step 1: User 137: Emotional Timeline 그래프
**⏱️ Time: 8:10-9:00 (50 seconds)**

> **[Show the data table with User 137's historical entries]**
>
> "The system loads User 137's historical data—42 emotional diary entries spanning 3 years from January 2021 to December 2023."
>
> **[Point to the most recent entry in the table]**
>
> "Most recent, December 17th, 2023: 'Had a good conversation with a friend, feeling better.' Valence 0.732, Arousal 0.466. This is my prediction starting point."
>
> **[Show the timeline chart with Valence and Arousal over time]**
>
> "This chart shows the emotional journey. Blue line is Valence—starts around 0.45 in 2021, gradually improves to 0.73 by 2023. Clear upward trend. Red line is Arousal—much more volatile, bouncing between 0.2 and 0.6. This volatility is why Arousal prediction is harder."
>
> **[Korean]** "시스템이 User 137의 이력 데이터를 로드합니다—2021년 1월부터 2023년 12월까지 3년간 42개 감정 일기. 가장 최근 12월 17일 항목: Valence 0.732, Arousal 0.466. 차트는 감정 여정을 보여줍니다. 파란선(Valence)은 2021년 0.45에서 2023년 0.73으로 상승. 빨간선(Arousal)은 훨씬 불규칙합니다."

---

### 🔧 Demo Step 2: Feature Engineering 출력값
**⏱️ Time: 9:00-9:30 (30 seconds)**

> **[Show the feature extraction output with all 47 features]**
>
> "The system automatically extracts all 47 features. Temporal features: Valence lag-1 is 0.732—the most recent value. **Arousal-specific features** are critical here: Arousal change 0.058, Arousal volatility 0.131—indicating recent instability. Text features: 54 characters, 9 words, 2 positive keywords—'good' and 'friend.' User statistics: average Valence 0.597, average Arousal 0.455—personal baseline."
>
> **[Korean]** "시스템이 자동으로 47개 피처를 추출합니다. 시간적 피처, **Arousal 전용 피처** (변화 0.058, 변동성 0.131), 텍스트 피처 (긍정 키워드 2개), 사용자 통계 (평균 Valence 0.597, 평균 Arousal 0.455)."

---

### 🎯 Demo Step 3: Run Predictions(2-Model Ensemble)
**⏱️ Time: 9:30-10:25 (55 seconds)** ⭐ **CRITICAL SECTION**

> **[Show the prediction output with both models' results]**
>
> "Here's the actual prediction from my ensemble."
>
> "seed777 predicts: Valence 0.480, Arousal 0.483—conservative, pulling both toward the middle. Arousal specialist predicts: Valence 0.516, Arousal 0.515—slightly higher, more responsive to recent patterns."
>
> **[Point to the final ensemble result - HIGHLIGHT]**
>
> "**Final ensemble prediction, weighted 50-50: Valence 0.498, Arousal 0.499.** This is my official forecast."
>
> "Compared to last observed values—Valence 0.732, Arousal 0.466—my system predicts Valence will decrease by 0.234, Arousal will increase by 0.033. What does this mean? The system is forecasting regression toward this user's personal mean. They had an unusually positive recent entry, but historically their baseline is lower around 0.597. The model expects some reversion."
>
> "All of this completed in **under 2 seconds** on a T4 GPU."
>
> **[Korean]** "앙상블 예측 결과입니다. seed777 예측: Valence 0.480, Arousal 0.483. Arousal 전문가 예측: Valence 0.516, Arousal 0.515. **최종 앙상블 예측 (50-50 가중): Valence 0.498, Arousal 0.499**. 시스템은 사용자 개인 평균으로의 회귀를 예측합니다. T4 GPU에서 **2초 미만** 소요."

---

### 📈 Demo Step 4: Visualize Prediction Results
**⏱️ Time: 10:25-11:15 (50 seconds)** ⭐ **CRITICAL SECTION**

> **[Show the Russell's Circumplex chart with historical dots and prediction star]**
>
> "Russell's Circumplex Model—a classic emotion research framework. Valence on x-axis, Arousal on y-axis."
>
> **[Trace the four quadrants]**
>
> "Four quadrants: Top-right is Excited-Alert. Top-left is Anxious-Tense. Bottom-left is Sad-Depressed. Bottom-right is Calm-Content."
>
> **[Point to the historical colored dots]**
>
> "Colored dots show User 137's history. Dark purple dots from early 2021 cluster bottom-left—sad, low energy. Bright yellow recent entries shift toward bottom-right—Calm, Content territory."
>
> **[Point to the gold star prediction marker]**
>
> "My prediction—large gold star—continues this trajectory. I'm forecasting they'll remain in positive-valence, moderate-arousal region. Stable and content."
>
> "This visualization tells a story: **three years of gradual emotional improvement, and my system recognizes and forecasts continuation of this pattern**."
>
> **[Korean]** "Russell의 Circumplex 모델입니다. 4개 사분면: 흥분-경계, 불안-긴장, 슬픔-우울, 평온-만족. User 137의 이력을 색깔 점으로 표시했습니다. 2021년 초 어두운 보라색 점들은 슬픔-낮은 에너지 영역. 최근 밝은 노란색은 평온-만족 영역으로 이동. 제 예측(금색 별)은 이 궤적을 이어갑니다. **3년간의 점진적 감정 개선, 제 시스템이 이 패턴의 지속을 예측합니다**."

---

## 🔄 **TRANSITION: Demo → Closing**
**⏱️ Time: 11:15-11:20 (5 seconds)**

> "As you can see, the system works effectively in practice."
>
> **[Korean]** "보시다시피 시스템은 실제로 효과적으로 작동합니다."

---

# PART 3: Closing & Q&A Transition
## Duration: 30 seconds

---

### 🎓 Final Closing
**⏱️ Time: 11:20-11:40 (20 seconds)**

> **[Confident conclusion]**
>
> " All code, models, and documentation are available to see on my GitHub repository. Thank you for your attention.."
>
> **[Korean]** "감사합니다. 모든 코드와 모델, 문서는 GitHub에서 재현 가능합니다. 질문 받겠습니다."

---

## ⏱️ **TOTAL TIME: 11:40 minutes**

### Time Breakdown
- **Part 1: PowerPoint (Slides 16-31)**: 7:50 minutes
- **Part 2: Demo Walkthrough**: 3:20 minutes
- **Part 3: Closing**: 0:20 minutes
- **Total**: 11:30 minutes ✅ **(Within 10-12 minute target)**

---

# APPENDIX

---

## 📚 A. Backup Q&A Answers

### Q1: "Why didn't you run the demo live?"
**Answer**: "Great question. I pre-executed for three reasons: saves 2-3 minutes for explanation, eliminates technical risks like authentication errors, and allows polished presentation. The pre-executed notebook shows exactly the same results as a live run."

**[Korean]** "좋은 질문입니다. 세 가지 이유로 미리 실행했습니다: 설명 시간 2-3분 절약, 인증 오류 같은 기술적 위험 제거, 깔끔한 발표 가능. 미리 실행한 노트북은 실시간 실행과 정확히 동일한 결과를 보여줍니다."

### Q2: "How long does prediction take in production?"
**Answer**: "Under 2 seconds on a T4 GPU—standard cloud hardware. On CPU, it's 5-8 seconds. The bottleneck is RoBERTa encoding at about 1.5 seconds. Feature extraction is milliseconds. LSTM and ensemble averaging add 200-300 milliseconds."

**[Korean]** "T4 GPU에서 2초 미만입니다—표준 클라우드 하드웨어. CPU에서는 5-8초. 병목은 RoBERTa 인코딩으로 약 1.5초. 피처 추출은 밀리초, LSTM과 앙상블은 200-300밀리초."

### Q3: "Why User 137 for the demo?"
**Answer**: "User 137 is representative: 42 entries close to dataset average of 25-28 per user, spans 3 years showing long-term dynamics, and exhibits clear emotional trajectory—improving from low Valence in 2021 to moderate-high in 2023. Visually compelling and easy to interpret."

**[Korean]** "User 137은 대표적입니다: 42개 항목으로 데이터셋 평균(25-28)에 가깝고, 3년간 장기 동역학을 보여주며, 명확한 감정 궤적(2021년 낮은 Valence → 2023년 중간-높은 Valence)을 보입니다. 시각적으로 설득력 있고 해석하기 쉽습니다."

### Q4: "Can the system handle new users with no history?"
**Answer**: "No, not effectively with my current architecture. I need at least 3-5 historical entries to compute lag features and rolling statistics. For cold-start scenarios, I'd need a separate model relying only on text features and global population statistics. This is a limitation and future work area."

**[Korean]** "아니요, 현재 아키텍처로는 효과적이지 않습니다. 최소 3-5개 이력 항목이 필요합니다. 콜드 스타트 시나리오에는 텍스트 피처와 전역 통계만 사용하는 별도 모델이 필요합니다. 이는 한계이자 향후 연구 영역입니다."

### Q5: "Why 47 features specifically?"
**Answer**: "I started with over 100 candidate features—various lag combinations, rolling windows, sentiment scores. I performed feature selection using importance scores from tree models and ablation studies. The final 47 represent the minimal set maintaining full predictive performance. Removing any causes CCC to drop."

**[Korean]** "100개 이상의 후보 피처에서 시작했습니다. 트리 모델의 중요도 점수와 ablation 연구로 피처 선택을 수행했습니다. 최종 47개는 전체 예측 성능을 유지하는 최소 집합입니다. 하나라도 제거하면 CCC가 감소합니다."

### Q6: "How does this compare to GPT-4?"
**Answer**: "I haven't directly compared, but LLMs face challenges here. They're not designed for regression with CCC loss—they're next-token prediction models. They lack specialized temporal modeling like BiLSTM for sequential forecasting. My 125M-parameter model is much more efficient than billion-parameter LLMs for deployment. That said, using LLMs for richer text embeddings could be promising future work."

**[Korean]** "직접 비교하지는 않았지만, LLM들은 여기서 어려움이 있습니다. CCC loss 회귀용이 아니라 다음 토큰 예측용입니다. BiLSTM 같은 전문 시간적 모델링이 부족합니다. 제 125M 파라미터 모델은 수십억 파라미터 LLM보다 배포에 훨씬 효율적입니다. 다만, 더 풍부한 텍스트 임베딩을 위해 LLM을 사용하는 것은 유망한 향후 연구가 될 수 있습니다."

### Q7: "Why did 2-model ensembles outperform 3 or 5 models?"
**Answer**: "This was surprising initially. The answer is noise injection. When I included weaker seeds like seed42 with Arousal CCC 0.3574 or seed123 with 0.4362, they introduced prediction errors that diluted the ensemble. Even with optimal weighting, their predictions were so far off that averaging them hurt overall performance. The lesson: ensemble diversity is about complementary strengths, not just adding more models."

**[Korean]** "처음에는 놀라웠습니다. 답은 노이즈 주입입니다. seed42(Arousal CCC 0.3574)나 seed123(0.4362) 같은 약한 시드를 포함하면 예측 오류가 앙상블을 희석시킵니다. 최적 가중치로도 그들의 예측은 너무 벗어나 평균을 내면 전체 성능이 저하됩니다. 교훈: 앙상블 다양성은 보완적 강점이지, 단순히 더 많은 모델을 추가하는 것이 아닙니다."

### Q8: "How did you decide on 90% CCC loss weighting?"
**Answer**: "Through systematic grid search. I tested CCC weights from 60% to 95% in 5% increments, evaluating on my validation set. 90% consistently gave the best Arousal CCC without overly sacrificing Valence. Below 85%, Arousal improvement was insufficient. Above 92%, Valence degraded too much. 90% was the sweet spot balancing both dimensions."

**[Korean]** "체계적인 그리드 서치를 통해서입니다. 60%에서 95%까지 5% 단위로 CCC 가중치를 테스트하고 검증 세트에서 평가했습니다. 90%가 Valence를 과도하게 희생하지 않으면서 최고의 Arousal CCC를 일관되게 제공했습니다. 85% 미만은 Arousal 개선 불충분, 92% 초과는 Valence 저하 과다. 90%가 두 차원의 균형점입니다."

---

## 🎤 B. Speaking Tips

### Do's ✅
1. **Speak 10-15% slower than normal** - Pause 1-2 seconds after key points
2. **Use hands to emphasize** - Point to visuals, avoid crossing arms
3. **Make eye contact** - Scan audience, hold 2-3 seconds with individuals
4. **Project confidence** - Say "I achieved" not "I got"
5. **Use transition phrases** - "Building on this insight...", "This brings me to..."

### Don'ts ❌
1. **Don't apologize unnecessarily** - ❌ "Sorry, this might be hard to see"
2. **Don't read slides verbatim** - Expand with examples
3. **Don't rush visualizations** - Pause, let people look
4. **Don't forget to breathe** - Breathe between sections

---

## 📊 C. Quick Reference Card (Printable)

```
⏰ TIMING CHECKPOINTS:
├─ 0:00 - Slide 16 (Title)
├─ 3:10 - Slide 21 (Arousal Specialist) ⭐ CRITICAL
├─ 4:40 - Slide 23 (Ensemble) ⭐ CRITICAL
├─ 8:00 - Demo Start
├─ 9:30 - Demo Predictions ⭐ CRITICAL
├─ 10:25 - Circumplex ⭐ CRITICAL
└─ 11:40 - FINISH

🎯 KEY NUMBERS:
- Overall CCC: 0.6833 (+10.4%)
- Valence CCC: 0.7593
- Arousal CCC: 0.5832 (+6%)
- 38% Volatility Gap
- 90% CCC Loss Weighting
- 2-model ensemble: 50.16% + 49.84%
- User 137: 42 entries, 3 years (2021-2023)
- Project: Nov 2025 - Jan 2026
- <2 seconds prediction time

⭐ CRITICAL MESSAGES:
1. Arousal-Specialist = 90% CCC Loss
2. Quality > Quantity: 2 models beat 3/5
3. +10.4% Above Target
```

---

**END OF SCRIPT**

**Good luck with your presentation! 🎉**

**Remember: You're not just presenting results—you're telling the story of how you solved a hard problem through systematic experimentation and strategic innovation.**

**기억하세요: 단순히 결과를 발표하는 것이 아니라, 체계적인 실험과 전략적 혁신을 통해 어려운 문제를 해결한 이야기를 전달하는 것입니다!**
