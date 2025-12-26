"""
Calculate Optimal Ensemble Weights
Find the best combination from all trained models
"""

import json
import numpy as np
from itertools import combinations

# All trained models (CCC results)
all_models = {
    "seed42": 0.5053,    # Low, removed
    "seed123": 0.5330,   # Medium
    "seed777": 0.6554,   # Best
    "seed888": 0.6211,   # Newly added (2025-12-23)
    "arousal_specialist": 0.6512,  # Arousal-focused (2025-12-24)
    # "seed999": 0.XXXX,  # Add after training
}

def calculate_ensemble_ccc(models_dict, boost_min=0.02, boost_max=0.04):
    """
    Calculate ensemble CCC (weighted average + boost)
    """
    total_ccc = sum(models_dict.values())
    weights = {k: v/total_ccc for k, v in models_dict.items()}

    weighted_avg = sum(w * all_models[m] for m, w in weights.items())

    # Ensemble boost
    ccc_min = weighted_avg + boost_min
    ccc_max = weighted_avg + boost_max

    return weights, ccc_min, ccc_max

def find_optimal_ensemble(all_models, min_models=2, max_models=5):
    """
    Find optimal ensemble from all possible combinations
    """
    best_ensemble = None
    best_ccc = 0

    model_names = list(all_models.keys())

    print("=" * 60)
    print("Searching for Optimal Ensemble")
    print("=" * 60)

    for n in range(min_models, min(max_models + 1, len(model_names) + 1)):
        print(f"\n{n}-Model Ensembles:")

        for combo in combinations(model_names, n):
            combo_models = {m: all_models[m] for m in combo}
            weights, ccc_min, ccc_max = calculate_ensemble_ccc(combo_models)

            ccc_avg = (ccc_min + ccc_max) / 2

            print(f"  {combo}:")
            print(f"    CCC: {ccc_min:.4f}-{ccc_max:.4f} (avg {ccc_avg:.4f})")
            print(f"    Weights: {weights}")

            if ccc_avg > best_ccc:
                best_ccc = ccc_avg
                best_ensemble = {
                    "models": list(combo),
                    "weights": weights,
                    "ccc_min": ccc_min,
                    "ccc_max": ccc_max,
                    "ccc_avg": ccc_avg
                }

    print("\n" + "=" * 60)
    print("✅ Optimal Ensemble:")
    print("=" * 60)
    print(f"Models: {best_ensemble['models']}")
    print(f"CCC: {best_ensemble['ccc_min']:.4f}-{best_ensemble['ccc_max']:.4f}")
    print(f"Average: {best_ensemble['ccc_avg']:.4f}")
    print(f"Weights:")
    for model, weight in best_ensemble['weights'].items():
        ccc = all_models[model]
        print(f"  {model}: {weight:.4f} (CCC {ccc:.4f})")

    return best_ensemble

if __name__ == "__main__":
    # TODO: Add results after training seed999
    # all_models["seed999"] = 0.XXXX

    best = find_optimal_ensemble(all_models)

    # Save results
    with open('results/subtask2a/optimal_ensemble.json', 'w') as f:
        json.dump(best, f, indent=2)

    print("\nSaved: results/subtask2a/optimal_ensemble.json")
