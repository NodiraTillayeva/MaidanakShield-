#!/usr/bin/env python3
"""
MAIDANAK SENTINEL - Fast AI Training Script (Lightweight)
Quick demonstration of GPS attack detection classifier.

Run: python3 train_gps_attack_fast.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import json
import pickle
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path('/Users/diyora/Desktop/Airbus/ai_training_results')
OUTPUT_DIR.mkdir(exist_ok=True)

fs = 10e6           # 10 MHz sampling rate
duration_s = 1.0
N = int(fs * duration_s)

NUM_SAMPLES_PER_CLASS = 50  # Fast: 50 samples per class = 300 total
CLASS_NAMES = ['clean', 'jamming', 'chirp', 'spoofing', 'multipath', 'transition']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

print("=" * 80)
print("MAIDANAK SENTINEL - FAST AI MODEL TRAINING")
print("(Lightweight scikit-learn version for quick demonstration)")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Samples per class: {NUM_SAMPLES_PER_CLASS}")
print(f"  Total training samples: {NUM_SAMPLES_PER_CLASS * len(CLASS_NAMES)}")
print(f"  Training will be fast: ~2-5 seconds\n")

# ============================================================================
# SIMPLE SIGNAL GENERATION
# ============================================================================

def white_noise(N, power_db=-30):
    noise = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    return noise * np.sqrt(10**(power_db/10))

def gps_like_signal(N, freq_offset=0, snr_db=18):
    t = np.arange(N) / fs
    carrier = np.exp(1j * 2 * np.pi * freq_offset * t)
    code = 2 * np.random.randint(0, 2, N) - 1
    signal_out = carrier * code
    signal_out = signal_out / np.sqrt(np.mean(np.abs(signal_out)**2))
    noise = white_noise(N, power_db=-snr_db)
    return signal_out + noise

def broadband_jamming(N, snr_db=5):
    jamming = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    return jamming * np.sqrt(10**(snr_db/10) * 5)

def chirp_jamming(N, f_start=-4e6, f_end=4e6):
    t = np.arange(N) / fs
    chirp = np.exp(1j * 2 * np.pi * (f_start + (f_end - f_start) * (t/duration_s)) * t)
    chirp = chirp / np.sqrt(np.mean(np.abs(chirp)**2))
    return chirp + white_noise(N, -10)

def spoofing_signal(N, freq_offset=500e3, snr_db=18):
    t = np.arange(N) / fs
    carrier = np.exp(1j * 2 * np.pi * freq_offset * t)
    code = 2 * np.random.randint(0, 2, N) - 1
    sig = carrier * code
    sig = sig / np.sqrt(np.mean(np.abs(sig)**2))
    return sig + white_noise(N, power_db=-snr_db)

def multipath_signal(N, snr_db=15):
    main = gps_like_signal(N, freq_offset=0, snr_db=snr_db)
    delay_samples = int(100e-6 * fs)
    delayed = np.zeros(N, dtype=complex)
    if delay_samples < N:
        delayed[delay_samples:] = 0.5 * main[:-delay_samples]
    return main + delayed

def generate_signal(class_name):
    """Generate a signal for a given class"""
    if class_name == 'clean':
        return gps_like_signal(N, snr_db=20) + white_noise(N, -28)
    elif class_name == 'jamming':
        return gps_like_signal(N, snr_db=12) + broadband_jamming(N)
    elif class_name == 'chirp':
        return gps_like_signal(N, snr_db=12) + chirp_jamming(N)
    elif class_name == 'spoofing':
        return gps_like_signal(N, snr_db=15) + spoofing_signal(N)
    elif class_name == 'multipath':
        return multipath_signal(N, snr_db=15)
    elif class_name == 'transition':
        n1 = int(0.30 * N)
        n2 = int(0.40 * N)
        n3 = N - n1 - n2
        p1 = gps_like_signal(n1, snr_db=18) + white_noise(n1, -28)
        p2 = broadband_jamming(n2)
        p3 = gps_like_signal(n3, snr_db=18) + white_noise(n3, -28)
        return np.concatenate([p1, p2, p3])

def extract_features(signal_data):
    """Extract simple statistical features from signal"""
    # Time-domain features
    time_signal = np.abs(signal_data)
    f1 = np.mean(time_signal)
    f2 = np.std(time_signal)
    f3 = np.max(time_signal)
    f4 = np.var(time_signal)
    
    # Power spectral density (simple FFT)
    fft_result = np.fft.fft(signal_data)
    psd = np.abs(fft_result)**2 / N
    f5 = np.mean(psd)
    f6 = np.std(psd)
    f7 = np.max(psd)
    f8 = np.argmax(psd)  # Peak frequency index
    
    # Crest factor
    f9 = np.max(time_signal) / (np.mean(time_signal) + 1e-10)
    
    # Peak-to-average power ratio
    f10 = np.max(psd) / (np.mean(psd) + 1e-10)
    
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])

# ============================================================================
# GENERATE DATASET
# ============================================================================

print("Generating synthetic training data...")
print("-" * 80)

X = []
y = []

for class_idx, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name:12s} ... ", end='', flush=True)
    
    for _ in range(NUM_SAMPLES_PER_CLASS):
        sig = generate_signal(class_name)
        features = extract_features(sig)
        X.append(features)
        y.append(class_idx)
    
    print(f"✓ ({NUM_SAMPLES_PER_CLASS} samples)")

X = np.array(X)
y = np.array(y)

print(f"\nDataset created:")
print(f"  Total samples: {X.shape[0]}")
print(f"  Features per sample: {X.shape[1]}")
print(f"  Classes: {len(CLASS_NAMES)}")

# ============================================================================
# NORMALIZE & SPLIT
# ============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain/test split:")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Testing: {X_test.shape[0]} samples")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print(f"\nTraining Random Forest classifier...")
print(f"  Estimators: 50")
print(f"  Max depth: 10")

clf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

# ============================================================================
# EVALUATE
# ============================================================================

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nResults:")
print(f"  Training accuracy: {train_acc:.2%}")
print(f"  Test accuracy: {test_acc:.2%}")

print(f"\nDetailed classification report:")
print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))

# ============================================================================
# SAVE MODEL
# ============================================================================

model_path = OUTPUT_DIR / 'gps_attack_detector.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({'model': clf, 'scaler': scaler}, f)
print(f"✓ Model saved: {model_path}")

history_path = OUTPUT_DIR / 'training_history.json'
with open(history_path, 'w') as f:
    json.dump({
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'timestamp': datetime.now().isoformat(),
        'model_type': 'RandomForest'
    }, f, indent=2)
print(f"✓ History saved: {history_path}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted', fontweight='bold')
ax.set_ylabel('True', fontweight='bold')
ax.set_title('Confusion Matrix - GPS Attack Detection', fontweight='bold', fontsize=12)
plt.tight_layout()
cm_file = OUTPUT_DIR / 'confusion_matrix.png'
plt.savefig(cm_file, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved: {cm_file}")
plt.close()

# Per-class accuracy
per_class = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(CLASS_NAMES, per_class, color='steelblue')
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Per-Class Detection Accuracy', fontweight='bold', fontsize=12)
ax.set_ylim([0, 1.1])
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.0%}',
            ha='center', va='bottom', fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
acc_file = OUTPUT_DIR / 'per_class_accuracy.png'
plt.savefig(acc_file, dpi=300, bbox_inches='tight')
print(f"✓ Per-class accuracy saved: {acc_file}")
plt.close()

# Feature importance
importance = clf.feature_importances_
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(len(importance)), importance, color='steelblue')
ax.set_yticks(range(len(importance)))
ax.set_yticklabels([f'Feature {i}' for i in range(len(importance))])
ax.set_xlabel('Importance', fontweight='bold')
ax.set_title('Feature Importance in Random Forest Model', fontweight='bold', fontsize=12)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
imp_file = OUTPUT_DIR / 'feature_importance.png'
plt.savefig(imp_file, dpi=300, bbox_inches='tight')
print(f"✓ Feature importance saved: {imp_file}")
plt.close()

print(f"\n{'='*80}")
print(f"✓ TRAINING COMPLETE!")
print(f"{'='*80}")
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"\nNext step: Run inference script")
print(f"  python3 inference_gps_attack_fast.py")
