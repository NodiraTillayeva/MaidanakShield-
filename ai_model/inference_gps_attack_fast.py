#!/usr/bin/env python3
"""
MAIDANAK SENTINEL - Fast Inference Script
Load trained model and make predictions.

Run: python3 inference_gps_attack_fast.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

fs = 10e6
duration_s = 1.0
N = int(fs * duration_s)

CLASS_NAMES = ['clean', 'jamming', 'chirp', 'spoofing', 'multipath', 'transition']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

MODEL_PATH = Path('/Users/diyora/Desktop/Airbus/ai_training_results/gps_attack_detector.pkl')
OUTPUT_DIR = Path('/Users/diyora/Desktop/Airbus/ai_inference_results')
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("MAIDANAK SENTINEL - INFERENCE")
print("=" * 80)

# ============================================================================
# SIGNAL GENERATION
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
    time_signal = np.abs(signal_data)
    f1 = np.mean(time_signal)
    f2 = np.std(time_signal)
    f3 = np.max(time_signal)
    f4 = np.var(time_signal)
    
    fft_result = np.fft.fft(signal_data)
    psd = np.abs(fft_result)**2 / N
    f5 = np.mean(psd)
    f6 = np.std(psd)
    f7 = np.max(psd)
    f8 = np.argmax(psd)
    
    f9 = np.max(time_signal) / (np.mean(time_signal) + 1e-10)
    f10 = np.max(psd) / (np.mean(psd) + 1e-10)
    
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])

# ============================================================================
# LOAD MODEL & MAKE PREDICTIONS
# ============================================================================

if not MODEL_PATH.exists():
    print(f"✗ Model not found at: {MODEL_PATH}")
    print(f"  Run: python3 train_gps_attack_fast.py")
    exit(1)

with open(MODEL_PATH, 'rb') as f:
    data = pickle.load(f)
    clf = data['model']
    scaler = data['scaler']

print(f"✓ Model loaded from: {MODEL_PATH}\n")
print("-" * 80)
print("\nMaking predictions on test signals...\n")

for class_name in CLASS_NAMES:
    print(f"{class_name.upper():12s} ... ", end='', flush=True)
    
    # Generate signal
    sig = generate_signal(class_name)
    
    # Extract features
    features = extract_features(sig)
    features_scaled = scaler.transform([features])[0]
    
    # Predict
    pred = clf.predict([features_scaled])[0]
    pred_name = CLASS_NAMES[pred]
    probs = clf.predict_proba([features_scaled])[0]
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain
    ax = axes[0, 0]
    t = np.arange(N) / fs
    ax.plot(t, np.abs(sig), linewidth=0.5, color='steelblue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('RF Signal (Time Domain)')
    ax.grid(True, alpha=0.3)
    
    # Frequency domain
    ax = axes[0, 1]
    fft_result = np.fft.fft(sig)
    freqs = np.fft.fftfreq(N, 1/fs)
    psd = np.abs(fft_result)**2 / N
    mask = freqs >= 0
    ax.semilogy(freqs[mask] / 1e6, psd[mask], linewidth=1, color='orange')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Power Spectral Density')
    ax.set_xlim([0, 5])
    ax.grid(True, alpha=0.3)
    
    # Predictions
    ax = axes[1, 0]
    colors = ['green' if p == pred else 'lightgray' for p in range(len(CLASS_NAMES))]
    ax.barh(CLASS_NAMES, probs, color=colors)
    ax.set_xlabel('Probability')
    ax.set_title('Model Predictions')
    ax.set_xlim([0, 1])
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probs)):
        ax.text(prob + 0.02, i, f'{prob:.2f}', va='center')
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
    TRUE CLASS: {class_name.upper()}
    PREDICTED: {pred_name.upper()}
    CONFIDENCE: {probs[pred] * 100:.1f}%
    
    {'✓ CORRECT' if class_name == pred_name else '✗ INCORRECT'}
    """
    ax.text(0.1, 0.5, summary, fontsize=13, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'GPS Attack Detection - {class_name.upper()}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / f'prediction_{class_name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Pred: {pred_name:12s} Conf: {probs[pred]:.2%} ✓")

print("\n" + "-" * 80)
print(f"\n✓ All predictions complete!")
print(f"  Results saved to: {OUTPUT_DIR}\n")
