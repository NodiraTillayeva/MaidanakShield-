#!/usr/bin/env python3
"""
MAIDANAK SENTINEL - Trainable GPS Attack Detection AI Model (scikit-learn version)
Generates synthetic GPS signals, trains a classifier, and evaluates on test data.
No PyTorch required—uses scikit-learn RandomForest.

Run: python3 train_gps_attack_classifier_sklearn.py

Requires: numpy, scipy, matplotlib, scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from pathlib import Path
import json
from datetime import datetime
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path('/Users/diyora/Desktop/Airbus/ai_training_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# Signal generation parameters
GPS_L1 = 1575.42e6  # Hz
GPS_L2 = 1227.60e6  # Hz
fs = 10e6           # Sampling frequency (10 MHz - typical for SDR)
duration_s = 1.0    # 1 second per signal
N = int(fs * duration_s)

# Training parameters
NUM_SAMPLES_PER_CLASS = 300  # Total 1800 samples (300 × 6 classes)
NPERSEG = 256
NOVERLAP = 128

# Class labels
CLASS_NAMES = ['clean', 'jamming', 'chirp', 'spoofing', 'multipath', 'transition']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

print("=" * 80)
print("MAIDANAK SENTINEL - GPS ATTACK DETECTION AI MODEL TRAINING")
print("(scikit-learn RandomForest version)")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Sampling rate: {fs/1e6:.0f} MHz")
print(f"  Signal duration: {duration_s} second(s)")
print(f"  Samples per class: {NUM_SAMPLES_PER_CLASS}")
print(f"  Total training samples: {NUM_SAMPLES_PER_CLASS * len(CLASS_NAMES)}")
print(f"  Spectrogram: STFT with nperseg={NPERSEG}, noverlap={NOVERLAP}\n")

# ============================================================================
# SIGNAL GENERATION FUNCTIONS
# ============================================================================

def white_noise(N, power_db=-30):
    """Generate white Gaussian noise"""
    noise = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    noise = noise * np.sqrt(10**(power_db/10))
    return noise

def gps_like_signal(N, freq_offset=0, snr_db=18):
    """Generate GPS-like narrowband signal"""
    t = np.arange(N) / fs
    carrier = np.exp(1j * 2 * np.pi * freq_offset * t)
    code = 2 * np.random.randint(0, 2, N) - 1
    signal = carrier * code
    signal = signal / np.sqrt(np.mean(np.abs(signal)**2))
    noise = white_noise(N, power_db=-snr_db)
    return signal + noise

def broadband_jamming(N, snr_db=5):
    """Generate broadband noise jamming"""
    jamming = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    return jamming * np.sqrt(10**(snr_db/10) * 5)

def chirp_jamming(N, f_start=-4e6, f_end=4e6, snr_db=5):
    """Generate frequency-swept jamming"""
    t = np.arange(N) / fs
    chirp = np.exp(1j * 2 * np.pi * (f_start + (f_end - f_start) * (t/duration_s)) * t)
    chirp = chirp / np.sqrt(np.mean(np.abs(chirp)**2))
    noise = white_noise(N, power_db=-snr_db)
    return (chirp + noise) * np.sqrt(10**(snr_db/10) * 3)

def spoofing_signal(N, freq_offset=500e3, snr_db=18):
    """Generate GPS spoofing attack"""
    t = np.arange(N) / fs
    carrier = np.exp(1j * 2 * np.pi * freq_offset * t)
    code = 2 * np.random.randint(0, 2, N) - 1
    signal = carrier * code
    signal = signal / np.sqrt(np.mean(np.abs(signal)**2))
    noise = white_noise(N, power_db=-snr_db)
    return (signal + noise) * np.sqrt(10**(snr_db/10) * 2)

def multipath_signal(N, snr_db=15):
    """Generate multipath interference"""
    main = gps_like_signal(N, freq_offset=0, snr_db=snr_db)
    delay_samples = int(100e-6 * fs)
    delayed = np.zeros(N, dtype=complex)
    if delay_samples < N:
        delayed[delay_samples:] = 0.5 * main[:-delay_samples]
    return main + delayed

def generate_signal(class_name, N, fs):
    """Generate a synthetic signal for a given class"""
    if class_name == 'clean':
        return gps_like_signal(N, freq_offset=0, snr_db=20) + white_noise(N, -28)
    elif class_name == 'jamming':
        return gps_like_signal(N, freq_offset=0, snr_db=12) + broadband_jamming(N)
    elif class_name == 'chirp':
        return gps_like_signal(N, freq_offset=0, snr_db=12) + chirp_jamming(N)
    elif class_name == 'spoofing':
        return gps_like_signal(N, freq_offset=0, snr_db=15) + spoofing_signal(N)
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
    else:
        raise ValueError(f"Unknown class: {class_name}")

def signal_to_spectrogram_features(signal, fs, nperseg=256, noverlap=128):
    """Convert signal to spectrogram and extract features"""
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    S = np.abs(Zxx)
    S_dB = 10 * np.log10(S + 1e-10)
    
    # Flatten 2D spectrogram into feature vector
    features = S_dB.flatten()
    
    # Add aggregate statistical features
    time_signal = np.abs(signal)
    features = np.concatenate([
        features,
        [
            np.mean(time_signal),
            np.std(time_signal),
            np.max(time_signal),
            np.var(S_dB),
            np.mean(S_dB),
            np.std(S_dB),
            np.max(S_dB),
        ]
    ])
    
    return features, f, t, S_dB

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    
    # Generate dataset
    print("\nGenerating synthetic training data...")
    
    X = []
    y = []
    
    for class_name in CLASS_NAMES:
        print(f"  Generating {NUM_SAMPLES_PER_CLASS} samples of {class_name}...", end='')
        
        for _ in range(NUM_SAMPLES_PER_CLASS):
            signal = generate_signal(class_name, N, fs)
            features, f, t, S_dB = signal_to_spectrogram_features(signal, fs, NPERSEG, NOVERLAP)
            
            X.append(features)
            y.append(CLASS_TO_IDX[class_name])
        
        print(" ✓")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    
    # Standardize features
    print(f"\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test
    print(f"\nSplitting into train (80%) and test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest
    print(f"\nTraining Random Forest classifier...")
    print(f"  Trees: 100")
    print(f"  Max depth: 20")
    print(f"  Min samples split: 5")
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"  Training accuracy: {train_accuracy:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}\n")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))
    
    # Save model and scaler
    model_path = OUTPUT_DIR / 'gps_attack_detector.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': clf, 'scaler': scaler}, f)
    print(f"✓ Model saved to: {model_path}")
    
    # Save training history
    history = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'class_names': CLASS_NAMES,
        'model_type': 'RandomForest',
        'n_estimators': 100,
        'n_features': int(X.shape[1])
    }
    history_path = OUTPUT_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to: {history_path}")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - GPS Attack Classification', fontsize=13, fontweight='bold')
    plt.tight_layout()
    cm_plot_path = OUTPUT_DIR / 'confusion_matrix.png'
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {cm_plot_path}")
    plt.close()
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(CLASS_NAMES, per_class_acc, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Detection Accuracy', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    per_class_path = OUTPUT_DIR / 'per_class_accuracy.png'
    plt.savefig(per_class_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class accuracy saved to: {per_class_path}")
    plt.close()
    
    # Feature importance
    feature_importance = clf.feature_importances_
    top_indices = np.argsort(feature_importance)[-15:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_indices)), feature_importance[top_indices], color='steelblue')
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([f'Feature {i}' for i in top_indices])
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Most Important Features', fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    importance_path = OUTPUT_DIR / 'feature_importance.png'
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance saved to: {importance_path}")
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nKey metrics:")
    print(f"  Training accuracy: {train_accuracy:.2%}")
    print(f"  Test accuracy: {test_accuracy:.2%}")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
