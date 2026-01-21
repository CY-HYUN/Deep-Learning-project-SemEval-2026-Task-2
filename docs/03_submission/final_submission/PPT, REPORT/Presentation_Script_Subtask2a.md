# SemEval 2026 Task 2a Presentation Script
## 10-15 Minute Natural Speaking Guide (English + Korean)

**Total Duration**: 12-15 minutes
**Slides**: 21 total
**Style**: Natural, conversational flow

---

## [SLIDE 1] Title Slide (30 sec)

### English Script:
Good afternoon, everyone. My name is Hyun Chang-Yong, and today I'll be presenting my work on SemEval 2026 Task 2, Subtask 2a—Emotional State Change Forecasting. This project is all about predicting how people's emotions change over time, specifically looking at two dimensions: Valence, which is basically how pleasant or unpleasant someone feels, and Arousal, which measures their energy level. I conducted this research at Télécom SudParis over 15 months, from November 2024 through January 2026.

### Korean Script:
안녕하세요, 현창용입니다. 오늘은 SemEval 2026 Task 2의 Subtask 2a, 감정 상태 변화 예측에 대한 제 연구를 발표하겠습니다. 이 프로젝트는 사람들의 감정이 시간에 따라 어떻게 변화하는지 예측하는 것으로, 특히 두 가지 차원을 봅니다. Valence는 기본적으로 사람이 얼마나 유쾌하거나 불쾌한지를 나타내고, Arousal은 에너지 수준을 측정합니다. 이 연구는 Télécom SudParis에서 2024년 11월부터 2026년 1월까지 15개월 동안 진행했습니다.

---

## [SLIDE 2] Joint Project Overview (40 sec)

### English Script:
Let me give you some context first. The Longitudinal Affect Challenge is about modeling emotional shifts over time using self-reported data from people's everyday writings, collected from 2021 to 2024. The competition has two parts. Subtask 1, which my teammate Rostislav worked on, predicts emotions for different users based on common language patterns. Subtask 2a—my part—takes a different approach. Instead of comparing across users, I'm forecasting one person's future emotional state by learning from their own history. So it's really a time-series forecasting problem, but applied to human emotions.

### Korean Script:
먼저 전체 맥락을 설명하겠습니다. Longitudinal Affect Challenge는 2021년부터 2024년까지 사람들이 일상에서 쓴 글의 자기보고 데이터를 사용해서 시간에 따른 감정 변화를 모델링하는 거예요. 대회는 두 부분으로 나뉩니다. Subtask 1은 제 팀원 Rostislav가 담당했고, 공통된 언어 패턴을 기반으로 다른 사용자들의 감정을 예측합니다. 제가 맡은 Subtask 2a는 다른 접근을 해요. 사용자 간 비교가 아니라, 한 사람의 과거 이력을 학습해서 그 사람의 미래 감정 상태를 예측하는 겁니다. 결국 시계열 예측 문제인데, 인간 감정에 적용한 거죠.

---

## [SLIDE 3] Part II Introduction (20 sec)

### English Script:
Alright, so now let's get into the technical details of Subtask 2a. My approach was to optimize how the model learns from long sequences of emotional data by using hybrid architectures and some specialized techniques for different emotional dimensions. I'll walk you through all of this step by step.

### Korean Script:
자, 이제 Subtask 2a의 기술적 세부사항으로 들어가겠습니다. 제 접근법은 하이브리드 아키텍처와 감정 차원별로 특화된 기법을 사용해서 모델이 긴 감정 데이터 시퀀스로부터 학습하는 방식을 최적화하는 거였어요. 차근차근 설명하겠습니다.

---

## [SLIDE 4] Dataset Analysis (60 sec) ⭐

### English Script:
So let's talk about the data. I worked with 137 training users and 46 test users over this 15-month period, and the data had some really interesting characteristics.

Looking at Valence first—that's the pleasantness dimension—the distribution was pretty balanced. Mean of 2.1, standard deviation of 0.9 on a 0-to-4 scale. Nothing too surprising there.

But Arousal? That's where things got interesting. The mean was only 1.0 with a standard deviation of just 0.6. This lower variance tells me that people have a much harder time consistently reporting their energy levels. And that turned out to be a huge prediction challenge.

Here are the four key insights I found. First, the sampling was irregular—people didn't write entries at regular intervals, so I needed to handle those varying time gaps. Second, each person wrote about 25 entries over a 3-year period. Third—and this is crucial—Arousal shifts happened 38% less frequently than Valence shifts, but when they did happen, they were way more sporadic and unpredictable. This 38% gap became the defining problem of my entire research. And fourth, there was missing data scattered throughout, which I dealt with through careful feature engineering.

### Korean Script:
먼저 데이터에 대해 얘기해볼게요. 이 15개월 동안 훈련 사용자 137명과 테스트 사용자 46명을 다뤘는데, 데이터가 정말 흥미로운 특성을 가지고 있었어요.

먼저 Valence—유쾌함 차원—를 보면 분포가 꽤 균형잡혀 있었어요. 0-4 스케일에서 평균 2.1, 표준편차 0.9. 특별히 놀랄 건 없었죠.

그런데 Arousal은요? 여기서 재미있어집니다. 평균이 겨우 1.0이고 표준편차가 0.6밖에 안 됐어요. 이 낮은 분산이 의미하는 건, 사람들이 자신의 에너지 수준을 일관되게 보고하는 게 훨씬 어렵다는 거예요. 그리고 이게 엄청난 예측 도전이 됐습니다.

네 가지 주요 인사이트를 찾았어요. 첫째, 샘플링이 불규칙했어요—사람들이 일정한 간격으로 글을 쓰지 않아서 이 변화하는 시간 간격을 처리해야 했죠. 둘째, 각 사람이 3년에 걸쳐 약 25개 항목을 썼어요. 셋째—이게 핵심인데—Arousal 변화가 Valence 변화보다 38% 덜 빈번하게 일어났지만, 일어날 때는 훨씬 더 산발적이고 예측 불가능했어요. 이 38% 격차가 제 전체 연구의 핵심 문제가 됐습니다. 넷째, 곳곳에 결측 데이터가 있었는데, 이건 신중한 특징 공학으로 처리했어요.

---

## [SLIDE 5] Russell's Circumplex (30 sec)

### English Script:
Here's a visualization I created from my own prediction results. This shows Russell's Circumplex Model—a classic way to represent emotions in 2D space. The x-axis is Valence, the y-axis is Arousal, and each red dot is one of my model's predictions. You can see how the predictions spread across the four emotional quadrants: Excited when both are high, Tense when Arousal is high but Valence is low, Sad in the bottom-left, and Calm in the top-right. I made this chart myself using my model's actual output data to validate that the predictions were making sense psychologically.

### Korean Script:
여기 제 예측 결과로 직접 만든 시각화가 있습니다. Russell의 Circumplex 모델인데요, 감정을 2D 공간으로 표현하는 고전적인 방법이에요. x축이 Valence, y축이 Arousal이고, 빨간 점 하나하나가 제 모델의 예측입니다. 예측이 네 개의 감정 사분면에 걸쳐 어떻게 퍼져있는지 볼 수 있어요. 둘 다 높으면 Excited, Arousal은 높지만 Valence가 낮으면 Tense, 왼쪽 아래는 Sad, 오른쪽 위는 Calm이죠. 이 차트는 제가 제 모델의 실제 출력 데이터를 사용해서 예측이 심리학적으로 말이 되는지 검증하려고 직접 만든 겁니다.

---

## [SLIDE 6] Hybrid Model Architecture (60 sec) ⭐

### English Script:
Now let me show you the core architecture I designed. I built what I call a Semantic-Temporal Integration system. Basically, it captures both what the text means right now and how emotions have been evolving over time.

There are four main components. First, I used RoBERTa-base as the encoder—that's a 125-million parameter transformer that turns text into 768-dimensional embeddings. It understands the semantic meaning really well.

Second, I added a BiLSTM layer—bidirectional LSTM with two layers and 256 units each. This is what captures the temporal dynamics, how emotions transition from one state to another over time.

Third, there's a multi-head attention mechanism with 8 heads. This helps the model figure out which parts of the text sequence are most emotionally important.

And fourth—this was really critical—I used a dual-head output structure. Instead of one shared head for both dimensions, I separated Valence and Arousal into completely independent regression heads. This turned out to be key because it let me optimize each dimension differently, which I'll explain more when I talk about the Arousal specialist model.

### Korean Script:
이제 제가 설계한 핵심 아키텍처를 보여드릴게요. 제가 의미-시간 통합 시스템이라고 부르는 걸 만들었어요. 기본적으로 텍스트가 지금 무슨 의미인지와 감정이 시간에 따라 어떻게 진화해왔는지를 둘 다 포착합니다.

네 가지 주요 구성요소가 있어요. 첫째, 인코더로 RoBERTa-base를 썼어요—1억 2,500만 매개변수 트랜스포머로 텍스트를 768차원 임베딩으로 바꿔줍니다. 의미를 정말 잘 이해하죠.

둘째, BiLSTM 레이어를 추가했어요—각각 256 유닛을 가진 2개 레이어의 양방향 LSTM이에요. 이게 시간적 역학을 포착하는 부분이에요. 감정이 시간에 따라 한 상태에서 다른 상태로 어떻게 전환되는지를요.

셋째, 8개 헤드를 가진 멀티헤드 어텐션 메커니즘이 있어요. 이게 모델이 텍스트 시퀀스에서 어떤 부분이 감정적으로 가장 중요한지 파악하도록 도와줍니다.

넷째—이게 정말 중요했는데—이중 헤드 출력 구조를 썼어요. 두 차원에 공유 헤드 하나 쓰는 대신, Valence와 Arousal을 완전히 독립적인 회귀 헤드로 분리했어요. 이게 핵심이었던 게, 각 차원을 다르게 최적화할 수 있게 해줬거든요. Arousal specialist 모델 얘기할 때 더 설명하겠습니다.

---

## [SLIDE 7] Advanced 47-Dim Feature Taxonomy (50 sec)

### English Script:
Feature engineering was absolutely critical. I built a 47-dimensional feature system organized into three categories.

First, textual features—those 768 RoBERTa embeddings I just mentioned. These capture the deep semantic meaning of what people wrote.

Second, temporal features—20 dimensions worth. This includes lag features looking back 1, 2, and 3 timesteps, rolling averages over windows of 5 and 10, and here's what's important—three arousal-specific dynamics features. I added change, volatility, and acceleration metrics specifically for Arousal because I noticed it had that sporadic behavior I mentioned earlier. These turned out to be essential.

Third, personal features—29 dimensions. I created learnable user embeddings, 64 dimensions per user, combined with their historical statistics. This personalizes the model to each individual's emotional patterns.

One technical detail worth mentioning: I implemented dynamic dimension handling so the system could flexibly work with 863 features for standard models and 866 for the specialized ones. This made ensemble integration much smoother.

### Korean Script:
특징 공학이 절대적으로 중요했어요. 세 가지 범주로 구성된 47차원 특징 시스템을 만들었습니다.

첫째, 텍스트 특징—방금 말한 그 768개 RoBERTa 임베딩이요. 이게 사람들이 쓴 글의 깊은 의미를 포착해요.

둘째, 시간 특징—20차원 분량이에요. 1, 2, 3 타임스텝 뒤를 보는 지연 특징, 5와 10 윈도우의 이동 평균이 포함되고, 여기 중요한 게 있어요—세 개의 arousal 전용 역학 특징입니다. 변화, 변동성, 가속도 메트릭을 Arousal 전용으로 추가했어요. 왜냐하면 아까 말한 그 산발적인 행동을 발견했거든요. 이게 필수적이었던 걸로 밝혀졌어요.

셋째, 개인 특징—29차원이에요. 사용자당 64차원의 학습 가능한 사용자 임베딩을 만들고, 과거 통계와 결합했어요. 이게 모델을 각 개인의 감정 패턴에 맞춰 개인화해줍니다.

언급할 만한 기술적 디테일 하나: 동적 차원 처리를 구현해서 시스템이 표준 모델은 863 특징으로, 특화 모델은 866으로 유연하게 작동할 수 있게 했어요. 이게 앙상블 통합을 훨씬 부드럽게 만들었죠.

---

## [SLIDE 8] Feature Importance Visualization (25 sec)

### English Script:
I created this breakdown chart to analyze how much each feature category actually contributed to the model. As you can see, RoBERTa embeddings dominate at 65% of the importance—that makes sense, the text is the primary signal. Temporal features account for 20%, personal features 10%, and those specialized arousal features I added? Just 5%. But don't let that small percentage fool you—that 5% turned out to be absolutely critical for solving the arousal prediction problem. I generated this chart myself by analyzing feature importance scores from my trained models.

### Korean Script:
각 특징 범주가 모델에 실제로 얼마나 기여하는지 분석하려고 이 분해 차트를 직접 만들었어요. 보시다시피 RoBERTa 임베딩이 65%로 압도적이에요—당연하죠, 텍스트가 주요 신호니까요. 시간 특징이 20%, 개인 특징이 10%, 그리고 제가 추가한 그 특화된 arousal 특징은요? 겨우 5%예요. 하지만 이 작은 퍼센티지에 속지 마세요—그 5%가 arousal 예측 문제를 해결하는 데 절대적으로 중요했던 걸로 밝혀졌어요. 이 차트는 제가 훈련시킨 모델들의 특징 중요도 점수를 분석해서 직접 만든 겁니다.

---

## [SLIDE 9] The Arousal Prediction Bottleneck (70 sec) ⭐⭐

### English Script:
Okay, this is where my research really took a turn. When I analyzed my initial ensemble results, I found this massive failure point—the model was doing significantly worse on Arousal compared to Valence. And this wasn't just a small gap. It was systematic.

So I dug into the root causes and found two fundamental issues. First, subjective variance. Think about it—when you ask someone "how energetic do you feel right now?" versus "how happy do you feel?", which one is easier to answer consistently? People really struggle with defining their energy levels. It's way more subjective and inconsistent than mood.

Second, low variation in the data. Because arousal changes happen less frequently and they're more subtle, standard loss functions just ignored them. The optimization naturally gravitated toward the higher-variance Valence dimension and basically neglected Arousal altogether.

This 38% performance gap between the two dimensions became the core challenge I had to solve. The big question was: how do I make the model actually care about Arousal predictions when they're so much harder and less frequent?

### Korean Script:
자, 여기서 제 연구가 진짜 전환점을 맞았어요. 초기 앙상블 결과를 분석했을 때, 엄청난 실패 지점을 발견했어요—모델이 Valence에 비해 Arousal에서 훨씬 더 안 좋은 성능을 보였거든요. 그리고 이게 작은 격차가 아니었어요. 체계적이었죠.

그래서 근본 원인을 파고들었더니 두 가지 근본적인 문제를 발견했어요. 첫째, 주관적 분산이에요. 생각해보세요—누군가에게 "지금 얼마나 활기차게 느끼세요?" 대 "얼마나 행복하게 느끼세요?"를 물으면, 어느 게 일관되게 답하기 더 쉬울까요? 사람들은 자신의 에너지 수준을 정의하는 데 정말 어려움을 겪어요. 기분보다 훨씬 더 주관적이고 일관성이 없죠.

둘째, 데이터의 낮은 변동이에요. arousal 변화가 덜 빈번하게 일어나고 더 미묘하기 때문에, 표준 손실 함수가 그냥 무시해버렸어요. 최적화가 자연스럽게 더 높은 분산의 Valence 차원으로 끌려가고 Arousal을 완전히 무시했죠.

두 차원 사이의 이 38% 성능 격차가 제가 해결해야 할 핵심 도전이 됐어요. 큰 질문은 이거였죠: 훨씬 더 어렵고 덜 빈번한데, 어떻게 모델이 실제로 Arousal 예측에 관심을 갖게 만들까?

---

## [SLIDE 10] Innovation: Arousal-Specialist Model (80 sec) ⭐⭐⭐

### English Script:
To solve this bottleneck, I developed a dedicated Arousal-Specialist model with three key innovations.

First and most important—Loss Shift with 90% CCC weighting. I dramatically increased the weight on the Concordance Correlation Coefficient loss for Arousal from 70% up to 90%. Why CCC specifically? Because CCC measures agreement between predictions and actual values, not just average error. So this forces the model to match the actual arousal patterns instead of just approximating some average. This single change had the biggest impact.

Second, dimension-specific parameter optimization. Instead of trying to optimize Valence and Arousal together, I trained this model with parameters tuned exclusively for Arousal performance. Different learning rates, different dropout patterns, different regularization—everything optimized for that one dimension.

Third, dynamic weighted sampling. During training, I oversampled the high-change emotional moments—exactly where the arousal predictions were failing. The model got to see way more examples of these difficult cases.

The result? An absolute 6% improvement in Arousal CCC—from 0.5281 all the way up to 0.5832. This didn't just incrementally improve things. It fundamentally solved the arousal prediction problem I'd been struggling with.

### Korean Script:
이 병목 현상을 해결하기 위해, 세 가지 주요 혁신을 가진 전용 Arousal-Specialist 모델을 개발했어요.

첫째이자 가장 중요한 것—90% CCC 가중치를 사용한 손실 변경이에요. Arousal에 대한 Concordance Correlation Coefficient 손실 가중치를 70%에서 90%로 극적으로 올렸어요. 왜 특별히 CCC냐고요? CCC는 단순히 평균 오차가 아니라 예측과 실제 값 사이의 일치도를 측정하거든요. 그래서 모델이 단순히 평균을 근사하는 게 아니라 실제 arousal 패턴을 일치시키도록 강제해요. 이 단일 변경이 가장 큰 영향을 미쳤어요.

둘째, 차원별 매개변수 최적화예요. Valence와 Arousal을 같이 최적화하려고 시도하는 대신, 이 모델을 Arousal 성능만을 위해 조정된 매개변수로 훈련시켰어요. 다른 학습률, 다른 드롭아웃 패턴, 다른 정규화—모든 게 그 한 차원을 위해 최적화됐죠.

셋째, 동적 가중 샘플링이에요. 훈련 중에 고변화 감정 순간들을 과표본했어요—arousal 예측이 실패하는 바로 그 지점들이요. 모델이 이 어려운 사례들을 훨씬 더 많이 볼 수 있게 됐죠.

결과는요? Arousal CCC의 절대 6% 개선—0.5281에서 0.5832까지 올라갔어요. 이게 단순히 점진적으로 개선한 게 아니에요. 제가 씨름하던 arousal 예측 문제를 근본적으로 해결했어요.

---

## [SLIDE 11] Detailed Model Benchmarks (30 sec)

### English Script:
Here you can see the benchmarks for all five models I trained. seed777 is my strongest generalist model—overall CCC of 0.6554, really good at Valence with 0.7593. The arousal_specialist has a slightly lower overall CCC at 0.6512, but look at the Arousal CCC—0.5832, the highest of all models. This complementary strength profile is exactly what I wanted for combining them in an ensemble.

### Korean Script:
여기 제가 훈련시킨 다섯 개 모델 전체의 벤치마크를 볼 수 있어요. seed777이 제 가장 강력한 범용 모델이에요—전체 CCC 0.6554, Valence에서 0.7593으로 정말 좋아요. arousal_specialist는 전체 CCC가 0.6512로 약간 낮지만, Arousal CCC를 보세요—0.5832로 모든 모델 중 가장 높아요. 이 상호 보완적인 강점 프로필이 정확히 제가 앙상블로 결합하려고 원했던 거예요.

---

## [SLIDE 12] Detailed Model Comparison Chart (25 sec)

### English Script:
I put together this comparison chart to visualize all the results clearly. It shows Combined CCC, Valence CCC, and Arousal CCC across all five models plus the final ensemble. See how the 2-model ensemble clearly beats the 0.62 target line, hitting 0.6833? This chart really highlights how the arousal specialist brings that unique strength in the arousal dimension that the generalist models were missing. I created this visualization myself using my actual model performance data.

### Korean Script:
모든 결과를 명확하게 시각화하려고 이 비교 차트를 직접 만들었어요. 다섯 개 모델과 최종 앙상블의 Combined CCC, Valence CCC, Arousal CCC를 보여줍니다. 2-모델 앙상블이 0.62 목표선을 명확히 넘어서 0.6833을 달성한 거 보이시죠? 이 차트가 arousal specialist가 범용 모델들에게 없던 arousal 차원의 독특한 강점을 어떻게 가져오는지 정말 잘 보여줘요. 이 시각화는 제 실제 모델 성능 데이터를 사용해서 직접 만든 겁니다.

---

## [SLIDE 13] The "Quality-over-Quantity" Ensemble (70 sec) ⭐⭐

### English Script:
This is probably the most counterintuitive finding of my research—why 2 models outperform 3 or 5.

I ran extensive grid searches, testing about 5,000 different weight combinations across all possible model configurations. And the clear winner was combining my strongest generalist, seed777, with the dimension specialist.

The final weights are almost perfectly balanced: seed777 at 50.16%—this one masters Valence and baseline trends. And arousal_specialist at 49.84%—this one corrects the energy-prediction bias. This nearly 50-50 split tells me the models have truly complementary strengths. Neither dominates.

But here's the interesting part—when I tried 3-model or 5-model configurations, performance actually got worse. Why? Because adding weaker models like seed42 with CCC of 0.5053 or seed123 with 0.5330 just injected noise that degraded the overall result. More isn't better—this is quality over quantity in action.

I also tested fancy meta-learning approaches—Ridge regression, XGBoost stacking, all that. Everything underperformed compared to this simple weighted average. Sometimes simple really is better.

Final achievement: CCC of 0.6833, which beats the competition target of 0.62 by 10.4%. That's a 13% improvement over my initial 3-model baseline.

### Korean Script:
이게 아마 제 연구에서 가장 반직관적인 발견일 거예요—왜 2개 모델이 3개나 5개보다 나은가.

광범위한 그리드 검색을 실행했어요. 모든 가능한 모델 구성에 걸쳐 약 5,000개의 다른 가중치 조합을 테스트했죠. 그리고 명확한 승자는 제 가장 강력한 범용 모델인 seed777을 차원 전문가와 결합하는 거였어요.

최종 가중치가 거의 완벽하게 균형잡혀 있어요. seed777이 50.16%—이게 Valence와 기준 트렌드를 마스터해요. 그리고 arousal_specialist가 49.84%—이게 에너지 예측 편향을 수정하죠. 이 거의 50-50 분할이 말해주는 건 모델들이 진정으로 상호 보완적인 강점을 가지고 있다는 거예요. 어느 쪽도 지배하지 않아요.

근데 재미있는 부분이 있어요—3-모델이나 5-모델 구성을 시도했을 때, 성능이 실제로 더 나빠졌어요. 왜냐고요? seed42 같은 약한 모델들을 추가하면—CCC 0.5053이거든요—또는 seed123을 추가하면—0.5330인데—전체 결과를 저하시키는 노이즈만 주입됐어요. 더 많다고 더 좋은 게 아니에요—이게 양보다 질이 실제로 작동하는 거죠.

고급 메타 학습 접근법도 테스트했어요—Ridge 회귀, XGBoost 스태킹, 다 해봤어요. 전부 이 단순 가중 평균에 비해 성능이 떨어졌어요. 때로는 단순한 게 정말 더 나아요.

최종 성과: CCC 0.6833인데, 대회 목표인 0.62를 10.4%나 넘었어요. 제 초기 3-모델 기준선보다 13% 개선된 거죠.

---

## [SLIDE 14] Comprehensive Results Summary (35 sec)

### English Script:
Let me wrap up the numbers for you. I achieved a final Combined CCC of 0.6833, which is 10.4% above the target. To get there, I trained 5 models total across different random seeds and configurations, using about 10 GPU hours on Google Colab's A100. The final ensemble? Just 2 models—the generalist and the specialist—in near-perfect balance. This chart I made from my own results really demonstrates that focused, intelligent model design beats just throwing more models at the problem.

### Korean Script:
숫자를 정리해드릴게요. 최종 Combined CCC 0.6833을 달성했는데, 목표보다 10.4% 높아요. 여기까지 오기 위해, 다양한 랜덤 시드와 구성에 걸쳐 총 5개 모델을 훈련시켰고, Google Colab의 A100에서 약 10 GPU 시간을 썼어요. 최종 앙상블은요? 겨우 2개 모델—범용과 전문가—거의 완벽한 균형으로요. 제 실제 결과로 만든 이 차트가 집중적이고 지능적인 모델 설계가 단순히 더 많은 모델을 던지는 것보다 낫다는 걸 정말 잘 보여줘요.

---

## [SLIDE 15] Technical Stack & Infrastructure (20 sec)

### English Script:
Quick overview of the tech stack. PyTorch 2.0+ and Hugging Face Transformers for deep learning, using RoBERTa-base with 125 million parameters. Infrastructure-wise, I relied heavily on Google Colab Pro with A100 GPUs and mixed precision training. Standard data science tools like pandas and scikit-learn, plus Python 3.10 with Git for version control.

### Korean Script:
기술 스택 간단히 정리할게요. 딥러닝은 PyTorch 2.0+와 Hugging Face Transformers, 1억 2,500만 매개변수의 RoBERTa-base 썼어요. 인프라는 Google Colab Pro의 A100 GPU와 혼합 정밀도 훈련에 많이 의존했고요. pandas, scikit-learn 같은 표준 데이터 과학 도구들, 그리고 버전 관리용 Git 포함된 Python 3.10 썼어요.

---

## [SLIDE 16] Challenges & Solutions (50 sec)

### English Script:
I faced four major challenges. First, that 38% Arousal Gap—solved it with the Arousal-Specialized Model using 90% CCC loss weighting and weighted sampling for high-variation events. Second, dimension mismatch between models—I implemented dynamic dimension handling that automatically adjusts between 863 and 866 features at runtime. Third, ensemble noise from combining too many models—solved through the quality-over-quantity approach, cutting out weaker seeds that were diluting performance. Fourth, resource constraints—I maximized efficiency by leveraging A100 infrastructure with optimized batch processing and gradient accumulation.

### Korean Script:
네 가지 주요 도전 과제에 직면했어요. 첫째, 그 38% Arousal 격차—90% CCC 손실 가중치와 고변동 이벤트에 대한 가중 샘플링을 사용한 Arousal-Specialized 모델로 해결했어요. 둘째, 모델 간 차원 불일치—런타임에 863과 866 특징 사이를 자동으로 조정하는 동적 차원 처리를 구현했죠. 셋째, 너무 많은 모델을 결합해서 생긴 앙상블 노이즈—성능을 희석시키던 약한 시드를 잘라내는 양보다 질 접근법으로 해결했어요. 넷째, 리소스 제약—최적화된 배치 처리와 그래디언트 누적으로 A100 인프라를 활용해서 효율성을 극대화했어요.

---

## [SLIDE 17] Key Learnings & Insights (50 sec)

### English Script:
Three big takeaways from this project. Technically, dimension-specific optimization turned out to be 4.3% more powerful than generic multi-tasking. Loss engineering—that 90% CCC weighting—is absolutely critical for agreement metrics, not just minimizing error. From a research perspective, systematic multi-seed experimentation reduced variance by 12%. Random seed actually matters more than people think. Ablation studies showed temporal features contribute 6% to overall accuracy—every feature category has its role. From a management perspective, timeline estimation is ridiculously hard—this took 15 months when I initially planned 4 weeks. But thorough documentation was essential for reproducibility.

### Korean Script:
이 프로젝트에서 세 가지 큰 교훈이 있어요. 기술적으로, 차원별 최적화가 일반적인 다중 작업보다 4.3% 더 강력한 걸로 밝혀졌어요. 손실 공학—그 90% CCC 가중치—은 단순히 오류를 최소화하는 게 아니라 일치 메트릭에 절대적으로 중요해요. 연구 관점에서, 체계적인 다중 시드 실험이 분산을 12% 줄였어요. 랜덤 시드가 사람들이 생각하는 것보다 훨씬 더 중요해요. 제거 연구는 시간 특징이 전체 정확도에 6% 기여한다는 걸 보여줬어요—모든 특징 범주가 역할이 있죠. 관리 관점에서, 타임라인 추정은 말도 안 되게 어려워요—처음에 4주 계획했는데 15개월 걸렸거든요. 근데 철저한 문서화는 재현성에 필수적이었어요.

---

## [SLIDE 18] Conclusion (35 sec)

### English Script:
To wrap up: I exceeded the target by achieving CCC of 0.6833, beating the 0.62 benchmark by 10.4% through systematic iteration and intelligent design. The key innovation—that Arousal-Specialized Model—solved the 38% prediction gap by shifting loss weighting to 90% CCC. This whole project really proves that dimension-specific optimization beats one-size-fits-all approaches, and that focused engineering with deep problem analysis outperforms just blindly scaling up models. Here's the final performance summary I created from my actual results showing we hit all our targets.

### Korean Script:
정리하자면: CCC 0.6833을 달성해서 목표를 초과했어요. 체계적인 반복과 지능적인 설계로 0.62 벤치마크를 10.4% 넘었죠. 주요 혁신인—그 Arousal-Specialized 모델—이 손실 가중치를 90% CCC로 바꿔서 38% 예측 격차를 해결했어요. 이 전체 프로젝트가 정말 증명하는 건 차원별 최적화가 일괄 적용 접근법을 이기고, 깊은 문제 분석을 가진 집중적 엔지니어링이 단순히 모델을 맹목적으로 스케일업하는 것을 능가한다는 거예요. 여기 제 실제 결과로 만든 최종 성능 요약인데 모든 목표를 달성한 게 보여요.

---

## [SLIDE 19] Future Directions (25 sec)

### English Script:
Looking ahead, short-term I'm planning to test RoBERTa-large for an expected 2-3% gain and explore meta-learning for dynamic adaptation. Long-term, I want to integrate multimodal signals like audio and video for better energy prediction, develop causal modeling for direct text-to-emotion relationships, and optimize for real-time inference under 100 milliseconds on mobile devices.

### Korean Script:
앞으로 계획은, 단기적으로 2-3% 향상을 기대하며 RoBERTa-large를 테스트하고 동적 적응을 위한 메타 학습을 탐색할 거예요. 장기적으로는, 더 나은 에너지 예측을 위해 오디오와 비디오 같은 다중 모달 신호를 통합하고, 직접적인 텍스트-감정 관계를 위한 인과 모델링을 개발하고, 모바일 기기에서 100밀리초 미만의 실시간 추론을 위해 최적화하고 싶어요.

---

## [SLIDE 20] Project Lifecycle & Key Milestones (20 sec)

### English Script:
The project ran 15 months through six phases: November 2024 EDA and baseline, December grid search hitting 0.63 CCC, late December specialist development, December 24th discovering optimal weights at 0.6833, January 7th production pipeline for predictions, and January 13th final documentation.

### Korean Script:
프로젝트는 6단계에 걸쳐 15개월 진행됐어요. 2024년 11월 EDA와 기준선, 12월 그리드 검색으로 CCC 0.63 달성, 12월 말 전문가 개발, 12월 24일 0.6833에서 최적 가중치 발견, 1월 7일 예측용 프로덕션 파이프라인, 1월 13일 최종 문서화였어요.

---

## [SLIDE 21] Questions & Thank You (20 sec)

### English Script:
Thank you for your attention. I'm happy to answer any questions about the architecture, the arousal specialization approach, the ensemble optimization, or anything else. This was my work on Subtask 2a for SemEval 2026 Task 2 at Télécom SudParis. My colleague Rostislav handled Subtask 1.

### Korean Script:
경청해 주셔서 감사합니다. 아키텍처, arousal 특화 접근법, 앙상블 최적화, 또는 다른 어떤 것에 대한 질문도 기꺼이 답변하겠습니다. 이건 Télécom SudParis에서 SemEval 2026 Task 2의 Subtask 2a에 대한 제 작업이었어요. 제 동료 Rostislav가 Subtask 1을 담당했습니다.

---

## Timing Summary

**Total Speaking Time**: 12-15 minutes

**Time Allocation**:
- Introduction (Slides 1-3): 1.5 minutes
- Dataset & Architecture (Slides 4-8): 4.5 minutes
- **Core Innovation (Slides 9-10): 2.5 minutes** ⭐ Most Important
- Results & Ensemble (Slides 11-14): 3 minutes
- Technical & Insights (Slides 15-17): 2 minutes
- Conclusion & Future (Slides 18-21): 1.5 minutes

**Key Emphasis Points**:
- Slides 9-10: Arousal Bottleneck + Specialist Model (most detailed)
- Slides 4, 6, 13: Dataset, Architecture, Ensemble (secondary emphasis)
- Slides 5, 8, 12, 14, 18: Personal visualizations created from own data

---

## Presentation Tips

1. **Natural Flow**: Read as if telling a story to a colleague, not reciting
2. **Voice Modulation**: Emphasize "38%", "90% CCC", "0.6833", "+10.4%"
3. **Pauses**: Breathe naturally between slides, especially before Slide 10
4. **Personal Touch**: When showing self-made visualizations (Slides 5, 8, 12, 14, 18), mention "I created this from my actual results"
5. **Conversational Tone**: Use contractions ("I'm", "that's", "it's") for natural speech

---

**End of Script**

**Recommended Practice**: 3-4 rehearsals to achieve natural 12-15 minute delivery
