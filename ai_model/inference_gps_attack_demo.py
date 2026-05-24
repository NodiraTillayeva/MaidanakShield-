#!/usr/bin/env python3
"""
MAIDANAK SENTINEL - Inference Demo Script
Load trained model and make predictions on new synthetic attack scenarios.

Run: python3 inference_gps_attack_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from datetime import datetime

OUTPUT_DIR = Path('/Users/diyora/Desktop/Airbus/ai_inference_results')
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = Path('/Users/diyora/Desktop/Airbus/ai_training_results/gps_attack_detector.pkl')
CLASS_NAMES = ['clean', 'jamming', 'chirp', 'spoofing', 'multipath', 'transition']
FEATURE_DIM = 10

print("=" * 80)
print("MAIDANAK SENTINEL - INFERENCE DEMONSTRATION")
print("=" * 80)

# ============================================================================
# LOAD MODEL
# ============================================================================

if not MODEL_PATH.exists():
    print(f"\n✗ Model not found: {MODEL_PATH}")
    print(f"  Please run: python3 train_gps_attack_demo.py")
    exit(1)

with open(MODEL_PATH, 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']

print(f"\n✓ Model loaded successfully\n")
print("-" * 80)
print("\nMaking predictions on test attack scenarios...\n")

# ============================================================================
# INFERENCE ON EACH CLASS
# ============================================================================

np.random.seed(123)  # Different seed from training for truly "new" data

class_params = {
    'clean': {'mean': 0.0, 'std': 0.5, 'offset': 0},
    'jamming': {'mean': 0.0, 'std': 1.5, 'offset': 2},
    'chirp': {'mean': 0.0, 'std': 1.2, 'offset': 1.5},
    'spoofing': {'mean': 0.0, 'std': 0.7, 'offset': 1.2},
    'multipath': {'mean': 0.0, 'std': 0.8, 'offset': 0.8},
    'transition': {'mean': 0.0, 'std': 1.3, 'offset': 2.5},
}

predictions_summary = []

for true_idx, true_class in enumerate(CLASS_NAMES):
    
    # Generate synthetic signal features for this class
    params = class_params[true_class]
    features = np.random.normal(
        loc=params['mean'] + params['offset'],
        scale=params['std'],
        size=FEATURE_DIM
    )
    
    # Normalize using training scaler
    features_scaled = scaler.transform([features])[0]
    
    # Get prediction and probabilities
    pred_idx = model.predict([features_scaled])[0]
    pred_class = CLASS_NAMES[pred_idx]
    probs = model.predict_proba([features_scaled])[0]
    confidence = probs[pred_idx]
    
    is_correct = (true_class == pred_class)
    
    # Print to console
    result_symbol = "✓" if is_correct else "✗"
    print(f"[{result_symbol}] {true_class.upper():12s} → Predicted: {pred_class.upper():12s} "
          f"(confidence: {confidence:.2%})")
    
    predictions_summary.append({
        'true_class': true_class,
        'predicted_class': pred_class,
        'confidence': confidence,
        'probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        'correct': is_correct
    })
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # 1. Feature values
    ax = fig.add_subplot(gs[0, :])
    colors_feat = ['green' if f > 0 else 'red' for f in features_scaled]
    ax.bar(range(FEATURE_DIM), features_scaled, color=colors_feat, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Normalized Value', fontweight='bold')
    ax.set_xlabel('Feature Index', fontweight='bold')
    ax.set_title(f'Input Features for {true_class.upper()} Attack', fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 2. Prediction probabilities
    ax = fig.add_subplot(gs[1, :])
    colors_pred = ['#2ecc71' if c == pred_class else '#95a5a6' for c in CLASS_NAMES]
    bars = ax.barh(CLASS_NAMES, probs, color=colors_pred, edgecolor='black', linewidth=1)
    ax.set_xlabel('Probability', fontweight='bold')
    ax.set_title(f'Model Confidence Scores', fontweight='bold')
    ax.set_xlim([0, 1])
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob + 0.03, bar.get_y() + bar.get_height()/2,
                f'{prob:.2%}', va='center', fontweight='bold', fontsize=10)
    
    # 3. Confusion indicator
    ax = fig.add_subplot(gs[2, 0])
    ax.axis('off')
    result_text = "✓ CORRECT" if is_correct else "✗ MISCLASSIFIED"
    result_color = '#2ecc71' if is_correct else '#e74c3c'
    ax.text(0.5, 0.7, result_text, ha='center', fontsize=18, fontweight='bold',
            transform=ax.transAxes, color=result_color)
    ax.text(0.5, 0.4, f"True:      {true_class.upper()}\nPredicted: {pred_class.upper()}\nConfidence: {confidence:.1%}",
            ha='center', fontsize=11, family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=0.8))
    
    # 4. All probabilities text
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    prob_text = "Class Breakdown:\n" + "-" * 25 + "\n"
    for cls, prob in zip(CLASS_NAMES, probs):
        prob_text += f"{cls:12s}: {prob:6.2%}\n"
    ax.text(0.05, 0.95, prob_text, transform=ax.transAxes, fontsize=10,
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=0.8))
    
    # Overall title
    fig.suptitle(f'GPS Attack Detection - {true_class.upper()} Scenario',
                 fontsize=15, fontweight='bold', y=0.98)
    
    # Save figure
    output_file = OUTPUT_DIR / f'prediction_{true_class}.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

print("\n" + "-" * 80)

# ========================================================================
# SUMMARY STATISTICS
# ========================================================================

correct_count = sum(1 for p in predictions_summary if p['correct'])
total_count = len(predictions_summary)
overall_accuracy = correct_count / total_count

print(f"\nOVERALL ACCURACY: {correct_count}/{total_count} ({overall_accuracy:.1%})")
print(f"\nDetailed Results:")
for pred in predictions_summary:
    status = "✓" if pred['correct'] else "✗"
    print(f"  {status} {pred['true_class']:12s} → {pred['predicted_class']:12s} ({pred['confidence']:.2%})")

print(f"\n{'='*80}")
print(f"✓ INFERENCE COMPLETE - {total_count} SCENARIOS EVALUATED")
print(f"{'='*80}")
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"  6 spectrogram + prediction visualizations generated")
print(f"\nUsage for presentation:")
print(f"  1. Show prediction_spoofing.png as main example (hardest to detect)")
print(f"  2. Show prediction_jamming.png (easiest to detect)")
print(f"  3. Show prediction_transition.png (real-world scenario)")
print(f"  4. Mention overall accuracy: {overall_accuracy:.1%}\n")
