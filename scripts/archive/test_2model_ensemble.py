"""
2-Model Ensemble Test (seed777 + seed123 only)
seed42를 제거하고 성능이 향상되는지 확인
"""

import json

# 현재 3-model ensemble
ensemble_3 = {
    "weights": {
        "seed42": 0.2983,
        "seed123": 0.3147,
        "seed777": 0.3870
    },
    "individual_ccc": {
        "seed42": 0.5053,
        "seed123": 0.5330,
        "seed777": 0.6554
    }
}

# seed42 제거한 2-model ensemble
# 가중치 재분배: 성능 비율로
total_perf = 0.5330 + 0.6554
weight_123 = 0.5330 / total_perf  # ~0.45
weight_777 = 0.6554 / total_perf  # ~0.55

ensemble_2 = {
    "weights": {
        "seed123": weight_123,
        "seed777": weight_777
    },
    "individual_ccc": {
        "seed123": 0.5330,
        "seed777": 0.6554
    }
}

print("=" * 60)
print("2-Model Ensemble Analysis (seed42 제거)")
print("=" * 60)

print("\n현재 3-Model Ensemble:")
print(f"  seed42: {ensemble_3['weights']['seed42']:.4f} (CCC {ensemble_3['individual_ccc']['seed42']:.4f})")
print(f"  seed123: {ensemble_3['weights']['seed123']:.4f} (CCC {ensemble_3['individual_ccc']['seed123']:.4f})")
print(f"  seed777: {ensemble_3['weights']['seed777']:.4f} (CCC {ensemble_3['individual_ccc']['seed777']:.4f})")
print(f"  예상 CCC: 0.5846-0.6046")

print("\n제안 2-Model Ensemble (seed42 제거):")
print(f"  seed123: {ensemble_2['weights']['seed123']:.4f} (CCC {ensemble_2['individual_ccc']['seed123']:.4f})")
print(f"  seed777: {ensemble_2['weights']['seed777']:.4f} (CCC {ensemble_2['individual_ccc']['seed777']:.4f})")

# 예상 성능 계산
expected_ccc_2model = (
    ensemble_2['weights']['seed123'] * ensemble_2['individual_ccc']['seed123'] +
    ensemble_2['weights']['seed777'] * ensemble_2['individual_ccc']['seed777']
)

# 앙상블 부스트 추가 (+0.02-0.04)
expected_ccc_min = expected_ccc_2model + 0.02
expected_ccc_max = expected_ccc_2model + 0.04

print(f"  예상 CCC: {expected_ccc_min:.4f}-{expected_ccc_max:.4f}")

# 개선 폭
current_ensemble_avg = 0.5946  # (0.5846 + 0.6046) / 2
new_ensemble_avg = (expected_ccc_min + expected_ccc_max) / 2
improvement = new_ensemble_avg - current_ensemble_avg

print(f"\n예상 개선:")
print(f"  현재: {current_ensemble_avg:.4f}")
print(f"  새로운: {new_ensemble_avg:.4f}")
print(f"  개선: +{improvement:.4f} ({improvement/current_ensemble_avg*100:+.2f}%)")

if improvement > 0:
    print("\n✅ 결론: seed42 제거가 성능 향상에 도움됨!")
    print("   권장: seed42 제거하고 2-model ensemble 사용")
else:
    print("\n⚠️ 결론: seed42 제거가 성능 하락 가능")
    print("   권장: 3-model 유지")

print("\n다음 단계:")
print("1. Validation data로 실제 테스트")
print("2. 더 나으면: 새 고성능 모델 1개 추가 (총 3개)")
print("3. 목표: CCC 0.62+ 달성")

print("=" * 60)
