"""
Extract and regenerate visualizations from demo notebook
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# Create output directory
output_dir = Path('demo_visualizations')
output_dir.mkdir(exist_ok=True)

print("Creating demo visualizations...")
print("=" * 60)

# ============================================================================
# Visualization 1: User 137 Emotional Timeline
# ============================================================================
print("\n1. Generating User 137 Emotional Timeline...")

# Generate sample data (matching notebook)
np.random.seed(137)
n_entries = 42
dates = pd.date_range(start='2021-01-15', periods=n_entries, freq='26D')

# Valence: starts low, improves over time
valence_trend = np.linspace(0.45, 0.70, n_entries)
valence_noise = np.random.normal(0, 0.08, n_entries)
valence = np.clip(valence_trend + valence_noise, 0, 1)

# Arousal: more volatile
arousal_base = 0.45 + 0.15 * np.sin(np.linspace(0, 4*np.pi, n_entries))
arousal_noise = np.random.normal(0, 0.10, n_entries)
arousal = np.clip(arousal_base + arousal_noise, 0, 1)

user_data = pd.DataFrame({
    'timestamp': dates,
    'valence': valence,
    'arousal': arousal
})

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))

# Valence timeline
ax1.plot(user_data.index, user_data['valence'], marker='o', color='blue',
         label='Valence', linewidth=2, markersize=6)
ax1.axhline(y=user_data['valence'].mean(), color='blue', linestyle='--',
            alpha=0.5, label=f'Mean: {user_data["valence"].mean():.3f}')
ax1.fill_between(user_data.index, user_data['valence'], alpha=0.2, color='blue')
ax1.set_ylabel('Valence', fontsize=13, fontweight='bold')
ax1.set_title('User 137: Emotional Timeline (3 years, 42 entries)',
              fontsize=15, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Arousal timeline
ax2.plot(user_data.index, user_data['arousal'], marker='o', color='red',
         label='Arousal', linewidth=2, markersize=6)
ax2.axhline(y=user_data['arousal'].mean(), color='red', linestyle='--',
            alpha=0.5, label=f'Mean: {user_data["arousal"].mean():.3f}')
ax2.fill_between(user_data.index, user_data['arousal'], alpha=0.2, color='red')
ax2.set_xlabel('Entry Number', fontsize=13, fontweight='bold')
ax2.set_ylabel('Arousal', fontsize=13, fontweight='bold')
ax2.legend(loc='best', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(output_dir / '01_user137_emotional_timeline.png')
plt.close()
print("   ✓ Saved: 01_user137_emotional_timeline.png")

# ============================================================================
# Visualization 2: Prediction Results (Timeline + Circumplex)
# ============================================================================
print("\n2. Generating Prediction Results Visualization...")

# Mock predictions (from notebook output)
pred_valence = 0.498
pred_arousal = 0.499

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Timeline with forecast
ax1 = axes[0]
x = list(range(len(user_data))) + [len(user_data)]
valence_history = list(user_data['valence']) + [pred_valence]
arousal_history = list(user_data['arousal']) + [pred_arousal]

ax1.plot(x[:-1], valence_history[:-1], marker='o', color='blue',
         label='Valence (Historical)', linewidth=2.5, markersize=6)
ax1.plot(x[:-1], arousal_history[:-1], marker='o', color='red',
         label='Arousal (Historical)', linewidth=2.5, markersize=6)

# Predicted values
ax1.scatter(x[-1], valence_history[-1], s=300, color='blue', marker='*',
            edgecolors='black', linewidths=2.5, label='Valence (Predicted)', zorder=5)
ax1.scatter(x[-1], arousal_history[-1], s=300, color='red', marker='*',
            edgecolors='black', linewidths=2.5, label='Arousal (Predicted)', zorder=5)

# Vertical line
ax1.axvline(x=len(user_data)-0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(len(user_data)-0.3, 0.95, 'Forecast', rotation=90, va='top',
         fontsize=12, fontweight='bold', color='gray')

ax1.set_xlabel('Entry Number', fontsize=13, fontweight='bold')
ax1.set_ylabel('Emotion Value', fontsize=13, fontweight='bold')
ax1.set_title('Emotional Trajectory + Forecast', fontsize=15, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)

# Right plot: Russell's Circumplex
ax2 = axes[1]

# Historical points
scatter = ax2.scatter(user_data['valence'], user_data['arousal'],
                      c=range(len(user_data)), cmap='viridis', s=120,
                      alpha=0.7, label='Historical', edgecolors='black', linewidths=0.5)

# Predicted point
ax2.scatter(pred_valence, pred_arousal,
            s=400, color='gold', marker='*', edgecolors='black',
            linewidths=3, label='Predicted', zorder=5)

# Quadrant lines
ax2.axhline(y=0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax2.axvline(x=0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Quadrant labels
ax2.text(0.25, 0.75, 'High Arousal\nNegative Valence\n(Anxious, Tense)',
         ha='center', va='center', fontsize=10, alpha=0.6,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax2.text(0.75, 0.75, 'High Arousal\nPositive Valence\n(Excited, Alert)',
         ha='center', va='center', fontsize=10, alpha=0.6,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax2.text(0.25, 0.25, 'Low Arousal\nNegative Valence\n(Sad, Depressed)',
         ha='center', va='center', fontsize=10, alpha=0.6,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax2.text(0.75, 0.25, 'Low Arousal\nPositive Valence\n(Calm, Content)',
         ha='center', va='center', fontsize=10, alpha=0.6,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax2.set_xlabel('Valence (Negative ← → Positive)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Arousal (Low ← → High)', fontsize=13, fontweight='bold')
ax2.set_title("Russell's Circumplex Model", fontsize=15, fontweight='bold')
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Time Progression (Entry #)', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / '02_prediction_results_combined.png')
plt.close()
print("   ✓ Saved: 02_prediction_results_combined.png")

# ============================================================================
# Visualization 3: Model Contribution Analysis
# ============================================================================
print("\n3. Generating Model Contribution Bar Charts...")

# Model predictions (from notebook)
seed777_valence = 0.480
seed777_arousal = 0.483
arousal_spec_valence = 0.516
arousal_spec_arousal = 0.515
ensemble_valence = 0.498
ensemble_arousal = 0.499

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

models = ['seed777', 'arousal_specialist', 'Ensemble']
valence_vals = [seed777_valence, arousal_spec_valence, ensemble_valence]
arousal_vals = [seed777_arousal, arousal_spec_arousal, ensemble_arousal]

last_valence = user_data['valence'].iloc[-1]
last_arousal = user_data['arousal'].iloc[-1]

# Valence comparison
bars1 = ax1.bar(models, valence_vals, color=['steelblue', 'lightblue', 'gold'],
                edgecolor='black', linewidth=2)
ax1.axhline(y=last_valence, color='red', linestyle='--',
            linewidth=2, label='Last observed')
ax1.set_ylabel('Valence', fontsize=13, fontweight='bold')
ax1.set_title('Valence Predictions by Model', fontsize=14, fontweight='bold')
ax1.set_ylim(0.3, 0.9)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Value labels
for bar, val in zip(bars1, valence_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

# Arousal comparison
bars2 = ax2.bar(models, arousal_vals, color=['coral', 'lightcoral', 'gold'],
                edgecolor='black', linewidth=2)
ax2.axhline(y=last_arousal, color='red', linestyle='--',
            linewidth=2, label='Last observed')
ax2.set_ylabel('Arousal', fontsize=13, fontweight='bold')
ax2.set_title('Arousal Predictions by Model', fontsize=14, fontweight='bold')
ax2.set_ylim(0.2, 0.7)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# Value labels
for bar, val in zip(bars2, arousal_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '03_model_contribution_analysis.png')
plt.close()
print("   ✓ Saved: 03_model_contribution_analysis.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("✅ All visualizations generated successfully!")
print("="*60)
print(f"\nOutput directory: {output_dir.absolute()}")
print("\nGenerated files:")
print("  1. 01_user137_emotional_timeline.png")
print("  2. 02_prediction_results_combined.png")
print("  3. 03_model_contribution_analysis.png")
print("\n" + "="*60)
