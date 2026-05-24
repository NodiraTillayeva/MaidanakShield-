#!/usr/bin/env python3
"""
MAIDANAK SENTINEL - Trainable GPS Attack Detection AI Model
Generates synthetic training data, trains a CNN classifier, and evaluates on test data.
Fully runnable with fake data; no external datasets required.

Run: python3 train_gps_attack_classifier.py

Requires: numpy, scipy, matplotlib, torch, scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
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
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_TEST_SPLIT = 0.8
NUM_SAMPLES_PER_CLASS = 300  # Total 1800 samples (300 × 6 classes)

# Class labels
CLASS_NAMES = ['clean', 'jamming', 'chirp', 'spoofing', 'multipath', 'transition']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

print("=" * 80)
print("MAIDANAK SENTINEL - GPS ATTACK DETECTION AI MODEL TRAINING")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Sampling rate: {fs/1e6:.0f} MHz")
print(f"  Signal duration: {duration_s} second(s)")
print(f"  Samples per class: {NUM_SAMPLES_PER_CLASS}")
print(f"  Total training samples: {NUM_SAMPLES_PER_CLASS * len(CLASS_NAMES)}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}\n")

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
    """Convert signal to spectrogram-based features"""
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    S = np.abs(Zxx)
    S_dB = 10 * np.log10(S + 1e-10)
    return S_dB, f, t

def extract_time_features(signal):
    """Extract time-domain features from signal"""
    abs_signal = np.abs(signal)
    return np.array([
        np.mean(abs_signal),
        np.std(abs_signal),
        np.max(abs_signal),
        np.mean(np.abs(np.diff(abs_signal))),
        np.var(abs_signal)
    ])

# ============================================================================
# DATASET CLASS
# ============================================================================

class GPSAttackDataset(Dataset):
    """PyTorch dataset for GPS attack signals"""
    
    def __init__(self, class_names, samples_per_class, fs, N, nperseg=256, noverlap=128):
        self.signals = []
        self.labels = []
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.fs = fs
        
        print("\nGenerating synthetic training data...")
        for class_name in class_names:
            print(f"  Generating {samples_per_class} samples of {class_name}...", end='')
            for _ in range(samples_per_class):
                signal = generate_signal(class_name, N, fs)
                spec, _, _ = signal_to_spectrogram_features(signal, fs, nperseg, noverlap)
                
                # Normalize spectrogram
                spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-10)
                
                self.signals.append(spec)
                self.labels.append(CLASS_TO_IDX[class_name])
            print(" ✓")
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = torch.FloatTensor(self.signals[idx]).unsqueeze(0)  # Add channel dimension
        label = torch.LongTensor([self.labels[idx]])[0]
        return signal, label

# ============================================================================
# CNN MODEL
# ============================================================================

class GPSAttackCNN(nn.Module):
    """Convolutional Neural Network for GPS attack detection"""
    
    def __init__(self, num_classes=6):
        super(GPSAttackCNN, self).__init__()
        
        # Input: (batch, 1, freq_bins, time_frames)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    """Train the CNN model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nTraining on {device}...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and return predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Generate dataset
    dataset = GPSAttackDataset(CLASS_NAMES, NUM_SAMPLES_PER_CLASS, fs, N)
    
    # Split into train and test
    train_size = int(TRAIN_TEST_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create validation loader from test set
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = GPSAttackCNN(num_classes=len(CLASS_NAMES)).to(device)
    print(f"\nModel architecture:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device
    )
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_preds, test_labels, test_probs = evaluate_model(model, val_loader, device)
    test_accuracy = np.mean(test_preds == test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}\n")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    
    # Save model
    model_path = OUTPUT_DIR / 'gps_attack_detector.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save training history
    history = {
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'val_accuracies': [float(x) for x in val_accuracies],
        'test_accuracy': float(test_accuracy),
        'class_names': CLASS_NAMES
    }
    history_path = OUTPUT_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to: {history_path}")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(val_accuracies, label='Validation Accuracy', linewidth=2, color='green')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Validation Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_plot_path = OUTPUT_DIR / 'training_curves.png'
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to: {loss_plot_path}")
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
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
    fig, ax = plt.subplots(figsize=(10, 6))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
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
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nKey metrics:")
    print(f"  Final test accuracy: {test_accuracy:.2%}")
    print(f"  Total trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
