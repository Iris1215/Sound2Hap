#!/usr/bin/env python3
"""
Perception-level audio-to-vibration translator after

  Lee & Choi, CHI 2013 – “Real-Time Perception-Level Translation from Audio Signals to Vibrotactile Effects”

Input  : mono 44 100 Hz, 16-bit WAV
Output : mono 8 000 Hz, 8-bit WAV
"""
import wave, struct, sys, math, argparse
import numpy as np
from scipy.signal import find_peaks
import soundfile as sf
import torch
from utils.normalization import normalize_audio
import os

# ---------------------------------------------------------------------------
# Constants
AUDIO_SR = 44100
VIB_SR = 8000 
FRAME_S = 4096
F1 = 175.0
F2 = 210.0
CR = 0.035 # 0.035 for games and 0.05 for movies
OR = 0.40 # 0.4 for both games and movies
CV = 1 # a simple scaling factor for Ra
CL = 0.1 # for music
OL = 3.8 # for music
c = 1.37

#  games and movies
C_fullband = 0.065
F_fullband = 6400 
# music
C_bass = 1.91
F_bass = 200 

# ---------------------------------------------------------------------------
# ISO-226:2003 60-phon equal-loudness contour
_ISO_FREQ  = np.array([25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                       800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300])

_ISO_SPL60 = np.array([104.23, 99.08, 94.18, 89.96, 85.94, 82.05, 78.65, 75.56, 72.47, 69.86, 67.53, 65.39, 
                       63.45, 62.05, 60.81, 59.89, 60.01, 62.15, 63.19, 59.96, 57.26, 56.42, 57.57, 60.89, 66.36])

def iso60phon(f):
    """60-phon SPL (dB) via linear interpolation."""
    return np.interp(f, _ISO_FREQ, _ISO_SPL60, left=_ISO_SPL60[0], right=_ISO_SPL60[-1])

# ---------------------------------------------------------------------------
# 1.  Per-frame auditory LOUDNESS, eq.(1)
def auditory_loudness(frame, content):
    """
      - For game/movie: full band up to 6400 Hz, C=0.065
      - For music: only up to 200 Hz, C=1.91
    """
    if content == "music":
        C_use = C_bass
        f_max = F_bass
    else:
        C_use = C_fullband
        f_max = F_fullband

    mag = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(frame.size, 1/AUDIO_SR)
    mask  = (freqs >= 25) & (freqs <= f_max)
    mag = mag[mask]
    freqs = freqs[mask]

    db = 20 * np.log10(c * mag + 1e-12)
    af = iso60phon(freqs)
    loudness = C_use * np.sum(db/af)
    return max(0.0, loudness)

# ---------------------------------------------------------------------------
# 2.  Per-frame auditory ROUGHNESS, eq.(2)
def auditory_roughness(frame, peak_db=-40.0): # -20 will make loud part stands out more
    mag = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(frame.size, 1/AUDIO_SR)

    mask = (freqs >= 25) & (freqs <= 6400)
    mag = mag[mask]
    freqs = freqs[mask]

    db = 20 * np.log10(mag + 1e-12)
    thresh = db.max() + peak_db
    peaks, _ = find_peaks(db, height=thresh)

    f = freqs[peaks]
    x = mag[peaks]
    R = 0.0

    for i in range(len(f)):
        for j in range(i+1, len(f)):
            f1, f2 = f[i], f[j]
            x1, x2 = x[i], x[j]
            xm, xM = min(x1,x2), max(x1,x2)
            fd = abs(f2 - f1)
            s = 0.24 / (0.0207*min(f1, f2) + 18.96)
            term = ((xm*xM)**0.1 / 2.0) * (2*xm/(xm+xM))**3.11
            R += term * (math.exp(-3.5*s*fd) - math.exp(-5.75*s*fd))
    return R

# ---------------------------------------------------------------------------
# 3.  Map (La, Ra) → (Iv, Rv)
def perceptual_targets(La, Ra, content):
    if content == "music":
        # eq.(8) for music
        Iv = CL*La - OL
    else:
        # eq.(7) for games and movies
        Iv = CR * math.sqrt(La) * (Ra**2) - OR     
    
    # eq.(9)
    Rv = CV * Ra                               
    return max(0, Iv), Rv

# ---------------------------------------------------------------------------
# 4.  Invert Iv, Rv
def amplitudes_from_percepts(Iv, Rv):
    if Iv <= 0.0:
        return 0.0, 0.0

    # eq.(5) – picking smaller valid root
    Rv_max = (801.0/113.0) + 0.529*Iv + 0.479
    Rv_adj = min(Rv, Rv_max)

    disc = max(0.0, 801.0 - 113.0*(Rv_adj - 0.529*Iv - 0.479))
    r1 = (28.3 + math.sqrt(disc)) / 56.3
    r2 = (28.3 - math.sqrt(disc)) / 56.3

    valid = [s for s in (r1, r2) if 0.0 <= s <= 1.0]
    
    if valid:
        S = min(valid)
        #print("valid")
    else:
        S = (28.3/56.3)  # Fallback: both were invalid → choose the closest Rv
        #print("not valid")

    # eq.(6)
    A = ((25.8*S**2 - 25.5*S + Rv_adj - 0.203) / 3.98)**2
    # any 0<γ<1 will lift small A
    # gamma = 0.5
    # A = A**gamma

    a2 = A * S
    a1 = A - a2
    return a1, a2

# ---------------------------------------------------------------------------
# 5.  Synthesize vibration segment (at 8 kHz) for one analysis block
def synth_vibration(a1, a2, n_samples):
    t = np.arange(n_samples) / VIB_SR
    segment = (a1 * np.sin(2*math.pi*F1*t) + a2 * np.sin(2*math.pi*F2*t))
    return segment

# ---------------------------------------------------------------------------
# 6.  WAV helpers
def read_wav_mono_44k(fname):
    wav_data, sr = sf.read(fname) 
    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=1)
    wav_tensor = torch.from_numpy(wav_data).float().unsqueeze(0) 
    wav_norm_tensor = normalize_audio(
        wav_tensor,
        normalize=True,
        strategy="peak",
        peak_clip_headroom_db=0,
        peak_normalize_db_clamp=0
    )
    return wav_norm_tensor.squeeze(0).numpy().astype("float32")

# ---------------------------------------------------------------------------
# 7.  Main pipeline
def process_file(in_wav, out_wav, overlap=0.0, content="game"):
    """
    Paper’s method: FRAME_S=4096 (≈93 ms), hop=4096 (no overlap), 
    C stays at 0.065, rectangular blocks.
    """
    audio = read_wav_mono_44k(in_wav) 
    HOP_S = FRAME_S 
    n_out_total = int(np.ceil(len(audio) * VIB_SR / AUDIO_SR))
    vib_full = np.zeros(n_out_total, dtype=np.float32)

    for start in range(0, len(audio), HOP_S):
        block = audio[start:start+FRAME_S]
        if block.size == 0:
            break
        if block.size < FRAME_S:
            # zero-pad last block
            block = np.pad(block, (0, FRAME_S - block.size), "constant")

        # 1) perceptual analysis
        La = auditory_loudness(block, content)
        Ra = auditory_roughness(block)
        Iv, Rv = perceptual_targets(La, Ra, content)
        a1, a2 = amplitudes_from_percepts(Iv, Rv)

        # print(f"Processing block: start={start}, La={La:.2f}, Ra={Ra:.2f}, " f"Iv={Iv:.2f}, Rv={Rv:.2f}, a1={a1:.4f}, a2={a2:.4f}")

        # 2) synth at 8 kHz
        n_out  = int(round(FRAME_S * VIB_SR / AUDIO_SR))
        vib_seg = synth_vibration(a1, a2, n_out)

        # Normalize the vibration segment (RMS-based)
        rms_seg = np.sqrt(np.mean(vib_seg ** 2) + 1e-12)
        vib_seg /= rms_seg * np.sqrt(2)

        # Apply a Hanning window for smooth transitions
        # window = np.hanning(len(vib_seg))
        # vib_seg *= window

        # 3) place contiguously
        out_start = int(round(start * VIB_SR / AUDIO_SR))
        out_end   = out_start + n_out
        if out_end > n_out_total:
            vib_full[out_start:] += vib_seg[: n_out_total - out_start]
        else:
            vib_full[out_start:out_end] += vib_seg

    # RMS-based normalization
    # rms = np.sqrt(np.mean(vib_full ** 2) + 1e-12)  # Add epsilon to avoid division by zero
    # vib_full /= rms * np.sqrt(2)  # Normalize RMS to √2 for consistent loudness
    vib_full = np.clip(vib_full, -1.0, 1.0)  # Clip for safety

    # # Adaptive amplification
    # rms_max = np.max(np.abs(vib_full))
    # rms_amplify = max(1.0, min(1.2, 1.0 / rms_max))  # Amplify if needed
    # vib_full *= rms_amplify

    # 5) write 8 kHz, 16-bit PCM
    sf.write(out_wav, vib_full, VIB_SR, subtype="PCM_16")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a haptic WAV from a single audio file using the perceptual mapping method."
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to the input WAV file."
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path for the generated haptic WAV file."
    )
    parser.add_argument(
        "--content_type", type=str, default="game",
        help="Content type: 'game' for games/movies, 'music' for music."
    )

    args = parser.parse_args()

    print(f"Processing: {args.input_file} → {args.output_file}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    process_file(
        in_wav=args.input_file,
        out_wav=args.output_file,
        overlap=0.5,
        content=args.content_type
    )
    print("Done.")