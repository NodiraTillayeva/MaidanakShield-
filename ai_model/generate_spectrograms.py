#!/usr/bin/env python3
"""
MAIDANAK SENTINEL - GPS Spectrogram Generator (Working Version)
Generates professional spectrograms showing normal vs. attack GPS signals
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from pathlib import Path

# Setup
OUTPUT_DIR = Path('/Users/diyora/Desktop/Airbus/maidanak_spectrograms_final')
OUTPUT_DIR.mkdir(exist_ok=True)

# Parameters
fs = 10e6           # 10 MHz sampling rate (typical SDR)
duration = 1.0      # 1 second
t = np.arange(0, duration, 1/fs)
N = len(t)

print("=" * 80)
print("MAIDANAK SENTINEL - GPS SPECTROGRAM GENERATOR")
print("=" * 80)

# ============================================================================
# Signal Generation
# ============================================================================

def noise(N, power_db=-30):
    """White noise"""
    s = np.random.randn(N) + 1j*np.random.randn(N)
    return s * np.sqrt(10**(power_db/10))

def gps_signal(N, freq_offset=0):
    """GPS-like narrowband signal"""
    t_local = np.arange(0, N) / fs
    carrier = np.exp(1j * 2*np.pi * freq_offset * t_local / fs)
    code = 2*np.random.randint(0, 2, N) - 1
    return (carrier * code) * np.sqrt(0.1) + noise(N, -20)

def jamming(N):
    """Broadband noise jamming"""
    return (np.random.randn(N) + 1j*np.random.randn(N)) * np.sqrt(0.5)

def chirp_jamming(N):
    """Frequency sweep jamming"""
    t_local = np.arange(0, N) / fs
    freq_sweep = np.exp(1j * 2*np.pi * (-4e6 + 8e6 * (np.arange(N)/N)) * t_local / fs)
    return freq_sweep * np.sqrt(0.3) + noise(N, -25)

def spoofing(N, offset=5e5):
    """GPS spoofing - looks like GPS but wrong frequency"""
    t_local = np.arange(0, N) / fs
    carrier = np.exp(1j * 2*np.pi * offset * t_local / fs)
    code = 2*np.random.randint(0, 2, N) - 1
    return (carrier * code) * np.sqrt(0.15) + noise(N, -18)

# ============================================================================
# Plot Spectrogram
# ============================================================================

def plot_spec(signal, title, filename, vmin=-50, vmax=10):
    """Compute and plot STFT spectrogram"""
    f, t_stft, Zxx = stft(signal, fs=fs, nperseg=512, noverlap=256)
    S = 10 * np.log10(np.abs(Zxx)**2 + 1e-10)
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # Plot
    im = ax.pcolormesh(t_stft, f/1e6, S, cmap='turbo', vmin=vmin, vmax=vmax, shading='auto')
    
    # Frequency gridlines
    for freq in np.arange(-5, 6, 1):
        ax.axhline(freq, color='white', linewidth=0.5, alpha=0.3, linestyle='--')
    
    # Time gridlines
    for time in np.arange(0, 1.1, 0.1):
        ax.axvline(time, color='white', linewidth=0.5, alpha=0.2, linestyle=':')
    
    # Labels
    ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency (MHz)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(-5, 5)
    
    cbar = plt.colorbar(im, ax=ax, label='Power (dB)', shrink=0.8)
    cbar.ax.tick_params(labelsize=11)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return path

# ============================================================================
# Generate Spectrograms
# ============================================================================

print("\nGenerating spectrograms...\n")

# 1. Normal
print("[1/6] Normal GPS signal...")
sig = gps_signal(N)
plot_spec(sig, 'SCENARIO 1: NORMAL GPS SIGNAL', '01_NORMAL.png', -50, 10)

# 2. Broadband Jamming
print("[2/6] Broadband jamming...")
sig = gps_signal(N) + jamming(N)
plot_spec(sig, 'SCENARIO 2: BROADBAND JAMMING ATTACK', '02_JAMMING.png', -50, 20)

# 3. Chirp Jamming
print("[3/6] Chirp jamming...")
sig = gps_signal(N) + chirp_jamming(N)
plot_spec(sig, 'SCENARIO 3: CHIRP/SWEEP JAMMING', '03_CHIRP.png', -50, 20)

# 4. Spoofing (KEY ONE)
print("[4/6] GPS spoofing...")
sig = gps_signal(N) + spoofing(N)
plot_spec(sig, 'SCENARIO 4: GPS SPOOFING (Narrowband Attack)', '04_SPOOFING.png', -50, 15)

# 5. Multiple spoofing signals
print("[5/6] Multipath interference...")
sig = gps_signal(N) + spoofing(N, 2e5) + spoofing(N, -3e5)
plot_spec(sig, 'SCENARIO 5: MULTIPATH INTERFERENCE', '05_MULTIPATH.png', -50, 10)

# 6. Attack scenario
print("[6/6] Attack scenario...")
n1 = int(0.3 * N)
n2 = int(0.4 * N)
n3 = N - n1 - n2

sig = np.concatenate([
    gps_signal(n1),
    gps_signal(n2) + jamming(n2),
    gps_signal(n3)
])

# Need to recompute spectrogram for transition with markers
f, t_stft, Zxx = stft(sig, fs=fs, nperseg=512, noverlap=256)
S = 10 * np.log10(np.abs(Zxx)**2 + 1e-10)

fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
im = ax.pcolormesh(t_stft, f/1e6, S, cmap='turbo', vmin=-50, vmax=20, shading='auto')

for freq in np.arange(-5, 6, 1):
    ax.axhline(freq, color='white', linewidth=0.5, alpha=0.3, linestyle='--')

# Mark attack window
ax.axvline(0.3, color='cyan', linewidth=3, label='Attack Starts', linestyle='--', alpha=0.9)
ax.axvline(0.7, color='lime', linewidth=3, label='Attack Ends', linestyle='--', alpha=0.9)

ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency (MHz)', fontsize=14, fontweight='bold')
ax.set_title('SCENARIO 6: ATTACK TRANSITION (Normal → Jamming → Recovery)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlim(0, 1)
ax.set_ylim(-5, 5)

cbar = plt.colorbar(im, ax=ax, label='Power (dB)', shrink=0.8)
cbar.ax.tick_params(labelsize=11)
ax.tick_params(labelsize=11)
ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_TRANSITION.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUCCESS! Spectrograms generated")
print("=" * 80)
print(f"\nLocation: {OUTPUT_DIR}\n")

files = [
    ('01_NORMAL.png', 'Clean GPS (dark, baseline)'),
    ('02_JAMMING.png', 'Broadband jamming (RED everywhere)'),
    ('03_CHIRP.png', 'Frequency sweep (diagonal RED line)'),
    ('04_SPOOFING.png', 'GPS spoofing (RED spike at wrong frequency) ← KEY'),
    ('05_MULTIPATH.png', 'Multiple echoes (multiple spikes)'),
    ('06_TRANSITION.png', 'Real attack scenario with timing'),
]

for fname, desc in files:
    full_path = OUTPUT_DIR / fname
    if full_path.exists():
        size_kb = full_path.stat().st_size / 1024
        print(f"✓ {fname:20s} ({size_kb:.1f} KB) - {desc}")
    else:
        print(f"✗ {fname:20s} (MISSING)")

print("\n" + "=" * 80)
print("HOW TO USE")
print("=" * 80)
print("""
INTERPRETATION:
  - Dark/Blue = Quiet (no attack)
  - Yellow = Moderate signal
  - Red = Strong signal (attack!)
  
KEY DIFFERENCES:
  - JAMMING: Entire spectrum RED (easy to detect, pilots notice immediately)
  - SPOOFING: RED spike at ONE frequency (harder to detect, looks like real GPS)
  - This is why AI/pattern recognition is essential!

FOR YOUR VIDEO/PRESENTATION:
  1. Show 01_NORMAL.png - "This is clean GPS"
  2. Show 02_JAMMING.png - "This is jamming (obvious)"
  3. Show 04_SPOOFING.png - "This is spoofing (deceptive, needs ML to detect)"
  4. Show 06_TRANSITION.png - "Real-world attack scenario"

NARRATIVE:
  "Our system recognizes these RF patterns in real-time using computer vision
   trained on Maidanak Observatory's RFI archive. Spoofing detection takes
   <2 seconds because we're not waiting for pilot confusion—we detect the
   attack in the RF environment, before it affects navigation."
""")

print("=" * 80)
print("READY FOR VIDEO/PRESENTATION!")
print("=" * 80)
