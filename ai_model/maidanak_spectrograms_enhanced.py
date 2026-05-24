#!/usr/bin/env python3
"""
MAIDANAK SENTINEL - Enhanced GPS Spectrogram Generator
Generates high-quality, visually striking spectrograms showing normal vs. attack scenarios
With clear frequency/time axes and professional formatting for presentations

Run: python3 maidanak_spectrograms_enhanced.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import stft, windows
import os
from pathlib import Path

# Configuration
OUTPUT_DIR = Path('/Users/diyora/Desktop/Airbus/maidanak_spectrograms_enhanced')
OUTPUT_DIR.mkdir(exist_ok=True)

# GPS Parameters
GPS_L1 = 1575.42e6  # Hz
GPS_L2 = 1227.60e6  # Hz
fs = 10e6           # Sampling frequency (10 MHz - typical for SDR)
duration = 1.0      # 1 second of signal
t = np.arange(0, duration, 1/fs)
N = len(t)

# STFT parameters for spectrograms
nperseg = 512       # FFT window size
noverlap = 256      # 50% overlap

print("=" * 80)
print("MAIDANAK SENTINEL - GPS SPECTROGRAM GENERATOR (ENHANCED)")
print("=" * 80)

# ============================================================================
# SIGNAL GENERATION FUNCTIONS
# ============================================================================

def generate_white_noise(N, power_db=-30):
    """Generate white Gaussian noise"""
    noise = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    noise_power = 10**(power_db/10)
    return noise * np.sqrt(noise_power)

def generate_gps_carrier(N, freq_offset=0, snr_db=15):
    """Generate GPS-like narrowband signal (L1 band)"""
    # GPS signal: narrowband carrier + spreading code
    carrier = np.exp(1j * 2 * np.pi * freq_offset * t / fs)
    
    # PRN code (pseudo-random sequence)
    code = 2 * np.random.randint(0, 2, N) - 1
    
    signal = carrier * code
    signal = signal / np.sqrt(np.mean(np.abs(signal)**2))  # Normalize
    
    # Add AWGN
    noise = generate_white_noise(N, power_db=-snr_db)
    return signal + noise

def generate_broadband_jamming(N, snr_db=5):
    """Generate broadband noise jamming attack"""
    # Continuous noise across entire spectrum
    jamming = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    jamming_power = 10**(snr_db/10)
    return jamming * np.sqrt(jamming_power * 5)  # Jamming is higher power

def generate_chirp_jamming(N, f_start=-4e6, f_end=4e6, snr_db=5):
    """Generate frequency-swept (chirp) jamming"""
    # Linear frequency sweep across band
    chirp = np.exp(1j * 2 * np.pi * (f_start + (f_end - f_start) * (np.arange(N)/N)) * t / fs)
    chirp = chirp / np.sqrt(np.mean(np.abs(chirp)**2))
    
    noise = generate_white_noise(N, power_db=-snr_db)
    jamming_power = 10**(snr_db/10)
    return (chirp + noise) * np.sqrt(jamming_power * 3)

def generate_spoofing_signal(N, freq_offset=5e5, snr_db=18):
    """Generate GPS spoofing attack (looks like GPS but wrong frequency)"""
    # Spoofed signal: narrowband like real GPS, but at different frequency
    carrier = np.exp(1j * 2 * np.pi * freq_offset * t / fs)
    code = 2 * np.random.randint(0, 2, N) - 1
    
    signal = carrier * code
    signal = signal / np.sqrt(np.mean(np.abs(signal)**2))
    
    # Spoofing is higher power (tries to overpower real signal)
    noise = generate_white_noise(N, power_db=-snr_db)
    spoofing_power = 10**(snr_db/10)
    return (signal + noise) * np.sqrt(spoofing_power * 2)

def generate_multipath(N, snr_db=15):
    """Generate multipath signal (delayed/weakened copies of GPS)"""
    main_signal = generate_gps_carrier(N, freq_offset=0, snr_db=snr_db)
    
    # Add delayed copies (multipath)
    delay_samples = 100
    delayed = np.zeros(N, dtype=complex)
    delayed[delay_samples:] = 0.5 * main_signal[:-delay_samples]  # 50% power, delayed
    
    return main_signal + delayed

# ============================================================================
# SPECTROGRAM COMPUTATION & VISUALIZATION
# ============================================================================

def compute_and_plot_spectrogram(signal, title, filename, vmin=-50, vmax=10):
    """Compute STFT and create publication-quality spectrogram"""
    
    # Compute STFT
    f, t_stft, Zxx = stft(signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    
    # Convert to dB scale
    S_dB = 10 * np.log10(np.abs(Zxx)**2 + 1e-10)
    
    # Create figure with larger size for clarity
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot spectrogram with enhanced colors
    im = ax.pcolormesh(t_stft, f/1e6, S_dB, shading='auto', cmap='turbo', 
                       vmin=vmin, vmax=vmax)
    
    # **FREQUENCY GRIDLINES** (this is what was missing!)
    freq_ticks = np.arange(-5, 6, 1)  # Every 1 MHz
    ax.set_yticks(freq_ticks)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5, color='white')
    
    # **TIME GRIDLINES**
    time_ticks = np.arange(0, duration + 0.1, 0.1)
    ax.set_xticks(time_ticks)
    ax.grid(True, axis='x', alpha=0.2, linestyle='--', linewidth=0.5, color='white')
    
    # Labels and title
    ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency (MHz)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Power (dB)')
    cbar.ax.tick_params(labelsize=11)
    
    # Tick sizes
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Set frequency range to be visible
    ax.set_ylim([-5, 5])
    ax.set_xlim([0, duration])
    
    # Tight layout
    plt.tight_layout()
    
    # Save with high DPI
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    
    plt.close()
    
    return filepath, S_dB

# ============================================================================
# GENERATE ALL SCENARIOS
# ============================================================================

print("\nGenerating spectrograms...\n")

# SCENARIO 1: NORMAL GPS SIGNAL (Clean)
print("[1/6] Normal GPS signal...")
signal_normal = generate_gps_carrier(N, freq_offset=0, snr_db=20) + generate_white_noise(N, -30)
compute_and_plot_spectrogram(
    signal_normal,
    'SCENARIO 1: NORMAL GPS SIGNAL (Clean)',
    '01_NORMAL.png',
    vmin=-50, vmax=10
)

# SCENARIO 2: BROADBAND JAMMING
print("[2/6] Broadband jamming attack...")
signal_jamming = generate_gps_carrier(N, freq_offset=0, snr_db=10) + generate_broadband_jamming(N)
compute_and_plot_spectrogram(
    signal_jamming,
    'SCENARIO 2: BROADBAND JAMMING ATTACK',
    '02_JAMMING.png',
    vmin=-50, vmax=20
)

# SCENARIO 3: CHIRP/SWEEP JAMMING
print("[3/6] Chirp jamming attack...")
signal_chirp = generate_gps_carrier(N, freq_offset=0, snr_db=10) + generate_chirp_jamming(N)
compute_and_plot_spectrogram(
    signal_chirp,
    'SCENARIO 3: CHIRP/SWEEP JAMMING ATTACK',
    '03_CHIRP_JAMMING.png',
    vmin=-50, vmax=20
)

# SCENARIO 4: GPS SPOOFING (THE KEY ONE)
print("[4/6] GPS spoofing attack...")
signal_spoofing = generate_gps_carrier(N, freq_offset=0, snr_db=15) + generate_spoofing_signal(N, freq_offset=5e5)
compute_and_plot_spectrogram(
    signal_spoofing,
    'SCENARIO 4: GPS SPOOFING (Narrowband Attack)',
    '04_SPOOFING.png',
    vmin=-50, vmax=15
)

# SCENARIO 5: MULTIPATH SIGNAL
print("[5/6] Multipath interference...")
signal_multipath = generate_multipath(N, snr_db=15)
compute_and_plot_spectrogram(
    signal_multipath,
    'SCENARIO 5: MULTIPATH INTERFERENCE',
    '05_MULTIPATH.png',
    vmin=-50, vmax=10
)

# SCENARIO 6: ATTACK TRANSITION (Normal → Jamming → Recovery)
print("[6/6] Attack transition scenario...")

# Create signal with phases
idx_normal = slice(0, int(0.3 * N))
idx_attack = slice(int(0.3 * N), int(0.7 * N))
idx_recovery = slice(int(0.7 * N), N)

signal_transition = np.zeros(N, dtype=complex)
signal_transition[idx_normal] = generate_gps_carrier(sum(1 for _ in range(len(t[idx_normal]))), snr_db=20)[idx_normal]
signal_transition[idx_attack] = generate_broadband_jamming(sum(1 for _ in range(len(t[idx_attack]))))
signal_transition[idx_recovery] = generate_gps_carrier(sum(1 for _ in range(len(t[idx_recovery]))), snr_db=20)[idx_recovery]

# Simpler transition: concatenate pre-generated signals
signal_normal_part = generate_gps_carrier(int(0.3*N), snr_db=20) + generate_white_noise(int(0.3*N), -30)
signal_attack_part = generate_broadband_jamming(int(0.4*N))
signal_recovery_part = generate_gps_carrier(int(0.3*N), snr_db=20) + generate_white_noise(int(0.3*N), -30)

signal_transition = np.concatenate([signal_normal_part, signal_attack_part, signal_recovery_part])

# Compute and plot
f, t_stft, Zxx = stft(signal_transition, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
S_dB = 10 * np.log10(np.abs(Zxx)**2 + 1e-10)

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.pcolormesh(t_stft, f/1e6, S_dB, shading='auto', cmap='turbo', vmin=-50, vmax=20)

# Gridlines
freq_ticks = np.arange(-5, 6, 1)
ax.set_yticks(freq_ticks)
ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5, color='white')

time_ticks = np.arange(0, duration + 0.1, 0.1)
ax.set_xticks(time_ticks)
ax.grid(True, axis='x', alpha=0.2, linestyle='--', linewidth=0.5, color='white')

# Mark attack window
ax.axvline(0.3, color='cyan', linestyle='--', linewidth=3, label='Attack Starts', alpha=0.8)
ax.axvline(0.7, color='lime', linestyle='--', linewidth=3, label='Attack Ends', alpha=0.8)

ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency (MHz)', fontsize=14, fontweight='bold')
ax.set_title('SCENARIO 6: ATTACK TRANSITION (Normal → Jamming → Recovery)', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim([-5, 5])
ax.set_xlim([0, duration])

cbar = plt.colorbar(im, ax=ax, label='Power (dB)')
cbar.ax.tick_params(labelsize=11)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_TRANSITION.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: 06_TRANSITION.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SPECTROGRAM GENERATION COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  ✓ 01_NORMAL.png          - Clean GPS signal (baseline)")
print("  ✓ 02_JAMMING.png         - Broadband noise jamming (high power everywhere)")
print("  ✓ 03_CHIRP_JAMMING.png   - Frequency sweep attack (diagonal pattern)")
print("  ✓ 04_SPOOFING.png        - GPS spoofing (narrowband, offset frequency)")
print("  ✓ 05_MULTIPATH.png       - Multipath interference (delayed copies)")
print("  ✓ 06_TRANSITION.png      - Real-world attack scenario")

print("\n" + "=" * 80)
print("HOW TO INTERPRET")
print("=" * 80)
print("""
AXES:
  X-axis (Time):      0 to 1 second
  Y-axis (Frequency): -5 to +5 MHz (relative to GPS L1)
  
COLORS:
  Dark Blue/Black = Quiet (-50 dB)
  Green           = Moderate signal (-20 dB)
  Yellow          = Strong signal (-10 dB)
  Red             = Very strong attack (0 dB)

KEY OBSERVATIONS:
  1. NORMAL: Mostly dark with occasional small bright spots
  2. JAMMING: Bright RED across entire frequency range (noise everywhere)
  3. CHIRP: Diagonal line (frequency changes over time)
  4. SPOOFING: Bright spike at ONE frequency (looks like real GPS)
  5. MULTIPATH: Multiple spikes (delayed copies)
  6. TRANSITION: Shows how attack appears/disappears in real-time
  
DETECTION HINT:
  - Jamming: Easy (entire spectrum lights up)
  - Spoofing: Harder (single bright spike, need pattern recognition to distinguish from real GPS)
  - This is why Maidanak's RFI training data is valuable!
""")

print("=" * 80)
print("Ready for video/presentation!")
print("=" * 80)
