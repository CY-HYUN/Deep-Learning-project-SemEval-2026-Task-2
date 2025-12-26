"""
Arousal-Focused Model Training

현재 Arousal 성능이 낮아서 (CCC 0.45)
Arousal에 특화된 모델 훈련

변경점:
1. Arousal loss weight 증가: 70% → 80% CCC
2. Arousal lag features 가중치 증가
3. Learning rate 조정
"""

# 기존 train_ensemble_subtask2a.py를 복사하고 다음만 변경:

# ===== CONFIGURATION =====
RANDOM_SEED = 888  # 또는 999
MODEL_SAVE_NAME = f'subtask2a_seed{RANDOM_SEED}_arousal_focused_best.pt'

# ===== LOSS WEIGHTS (변경!) =====
# Valence: 기존 유지
CCC_WEIGHT_V = 0.65
MSE_WEIGHT_V = 0.35

# Arousal: CCC 가중치 증가 (70% → 80%)
CCC_WEIGHT_A = 0.80  # ⭐ 기존 0.70에서 증가
MSE_WEIGHT_A = 0.20  # ⭐ 기존 0.30에서 감소

print('⭐ AROUSAL-FOCUSED MODEL')
print(f'Valence Loss: {CCC_WEIGHT_V*100:.0f}% CCC + {MSE_WEIGHT_V*100:.0f}% MSE')
print(f'Arousal Loss: {CCC_WEIGHT_A*100:.0f}% CCC + {MSE_WEIGHT_A*100:.0f}% MSE  ← INCREASED!')

# 나머지는 train_ensemble_subtask2a.py와 동일
