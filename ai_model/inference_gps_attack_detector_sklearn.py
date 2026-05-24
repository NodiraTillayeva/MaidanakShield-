#!/usr/bin/env python3
"""
MAIDANAK SENTINEL - GPS Attack Detection Inference (scikit-learn version)
Load a trained RandomForest model and make predictions on new signals.

Run: python3 inference_gps_attack_detector_sklearn.py

Requires: numpy, scipy, matplotlib, scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from pathlib import Path
import pickle

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = Path('/Users/diyora/Desktop/Airbus/ai_training_results/gps_attack_detector.pkl')
OUTPUT_DIR = Path('/Users/diyora/Desktop/Airbus/ai_inference_results')
OUTPUT_DIR.mkdir(exist_ok=True)

GPS_L1 = 1575.42e6
fs = 10e6
duration_s = 1.0
N = int(fs * duration_s)

NPERSEG = 256
NOVERLAP = 128

CLASS_NAMES = ['clean', 'jamming', 'chirp', 'spoofing', 'multipath', 'transition']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

print("=" * 80)
print("MAIDANAK SENTINEL - GPS ATTACK DETECTION INFERENCE")
print("(scikit-learn RandomForest version)")
print("=" * 80)

# ============================================================================
# SIGNAL GENERATION FUNCTIONS
# ============================================================================

def white_noise(N, power_db=-30):
    noise = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    return noise * np.sqrt(10**(power_db/10))

def gps_like_signal(N, freq_offset=0, snr_db=18):
    t = np.arange(N) / fs
    carrier = np.exp(1j * 2 * np.pi * freq_offset * t)
    code = 2 * np.random.randint(0, 2, N) - 1
    signal = carrier * code
    signal = signal / np.sqrt(np.mean(np.abs(signal)**2))
    noise = white_noise(N, power_db=-snr_db)
    return signal + noise

def broadband_jamming(N, snr_db=5):
    jamming = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    return jamming * np.sqrt(10**(snr_db/10) * 5)

def chirp_jamming(N, f_start=-4e6, f_end=4e6, snr_db=5):
    t = np.arange(N) / fs
    chirp = np.exp(1j * 2 * np.pi * (f_start + (f_end - f_start) * (t/duration_s)) * t)
    chirp = chirp / np.sqrt(np.mean(np.abs(chirp)**2))
    noise = white_noise(N, power_db=-snr_db)
    return (chirp + noise) * np.sqrt(10**(snr_db/10) * 3)

def spoofing_signal(N, freq_offset=500e3, snr_db=18):
    t = np.arange(N) / fs
    carrier = np.exp(1j * 2 * np.pi * freq_offset * t)
    code = 2 * np.random.randint(0, 2, N) - 1
    signal = carrier * code
    signal = signal / np.sqrt(np.mean(np.abs(signal)**2))
    noise = white_noise(N, power_db=-snr_db)
    return (signal + noise) * np.sqrt(10**(snr_db/10) * 2)

def multipath_signal(N, snr_db=15):
    main = gps_like_signal(N, freq_offset=0, snr_db=snr_db)
    delay_samples = int(100e-6 * fs)
    delayed = np.zeros(N, dtype=complex)
    if delay_samples < N:
        delayed[delay_samples:] = 0.5 * main[:-delay_samples]
    return main + delayed

def generate_signal(class_name, N, fs):
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

def signal_to_spectrogram_features(signal, fs, nperseg=256, noverlap=128):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    S = np.abs(Zxx)
    S_dB = 10 * np.log10(S + 1e-10)
    
    # Flatten spectrogram into feature vector
    features = S_dB.flatten()
    
    # Add statistical features
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
# INFERENCE CLASS
# ============================================================================

class GPSAttackDetector:
    """Wrapper class for GPS attack detection inference"""
    
    def __init__(self, model_path):
        if not model_path.exists():
            print(f"✗ Model file not found at: {model_path}")
            print(f"  Please run: python3 train_gps_attack_classifier_sklearn.py")
            exit(1)
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        
        print(f"✓ Model loaded from: {model_path}")
    
    def predict(self, signal, signal_name="Unknown"):
        """Make prediction on a signal"""
        # Convert to spectrogram features
        features, f, t, S_dB = signal_to_spectrogram_features(signal, fs, NPERSEG, NOVERLAP)
        
        # Standardize
        features_scaled = self.scaler.transform([features])[0]
        
        # Predict with probabilities
        pred_idx = self.model.predict([features_scaled])[0]
        probs = self.model.predict_proba([features_scaled])[0]
        
        pred_class = IDX_TO_CLASS[pred_idx]
        pred_confidence = probs[pred_idx]
        
        return {
            'signal_name': signal_name,
            'predicted_class': pred_class,
            'confidence': pred_confidence,
            'all_probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
            'spectrogram': S_dB,
            'frequencies': f,
            'times': t
        }

# ============================================================================
# DEMO PREDICTIONS
# ============================================================================

if __name__ == '__main__':
    
    # Load model
    detector = GPSAttackDetector(MODEL_PATH)
    
    print(f"\nGenerating test signals and making predictions...\n")
    print("-" * 80)
    
    # Test each class
    for class_name in CLASS_NAMES:
        print(f"\nTesting: {class_name.upper()}")
        
        # Generate signal
        signal = generate_signal(class_name, N, fs)
        
        # Make prediction
        result = detector.predict(signal, signal_name=class_name)
        
        # Print results
        print(f"  True class:        {result['signal_name']}")
        print(f"  Predicted class:   {result['predicted_class']}")
        print(f"  Confidence:        {result['confidence']:.4f}")
        
        # Show all probabilities
        print(f"  All probabilities:")
        for cls_name, prob in result['all_probabilities'].items():
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"    {cls_name:12s}  [{bar}] {prob:.4f}")
        
        # Visualize spectrogram
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.pcolormesh(result['times'], result['frequencies']/1e6, result['spectrogram'],
                           shading='auto', cmap='turbo')
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency (MHz)', fontsize=11, fontweight='bold')
        ax.set_ylim([-5, 5])
        ax.set_xlim([0, duration_s])
        
        # Add prediction to title
        title = f"{class_name.upper()} - Predicted: {result['predicted_class'].upper()} (conf: {result['confidence']:.3f})"
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        cb = plt.colorbar(im, ax=ax, label='Power (dB)')
        
        # Add text box with all predictions
        textstr = '\n'.join([f"{cls}: {prob:.3f}" for cls, prob in result['all_probabilities'].items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        # Save figure
        output_file = OUTPUT_DIR / f"prediction_{class_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        
        plt.close()
    
    print("\n" + "-" * 80)
    print(f"\n✓ All predictions complete!")
    print(f"  Results saved to: {OUTPUT_DIR}\n")
