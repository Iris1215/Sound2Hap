#!/usr/bin/env python3
# This script implements the frequency shifting algorithm described in the paper:
# @incollection{okazaki2015effect, title={The effect of frequency shifting on audio--tactile conversion for enriching musical experience}, author={Okazaki, Ryuta and Kuribayashi, Hidenori and Kajimoto, Hiroyuki}, booktitle={Haptic Interaction: Perception, Devices and Applications}, pages={45--51}, year={2015}, publisher={Springer} }

import os
import argparse
from pathlib import Path
from typing import Union
import numpy as np
import torch
import librosa, soundfile as sf
from scipy.signal import butter, lfilter
from scipy.signal import hilbert
from utils.normalization import normalize_audio

def _butter_bandpass(sr: int, center_hz: float = 250.0, q: float = 1.0, order: int = 4):
    bw = center_hz / q
    low_hz  = max(center_hz - bw/2, 1.0)
    high_hz = min(center_hz + bw/2, sr/2 - 1)
    wn = [low_hz/(sr/2), high_hz/(sr/2)]
    return butter(order, wn, btype="band")

def _butter_highpass(sr: int, cutoff_hz: float = 10.0, order: int = 2):
    wn = cutoff_hz / (sr/2)
    return butter(order, wn, btype="high")

def process_file(in_wav: Union[str, Path], out_wav: Union[str, Path], centre_hz: float = 250.0, q: float = 1.0) -> None:
    sr_out = 8000
    # Load (mono) using native sample rate
    y, sr = librosa.load(in_wav, sr=None, mono=True)

    # Peak-normalize
    wav_tensor = torch.from_numpy(y).float().unsqueeze(0)
    y_norm_t = normalize_audio(wav_tensor, normalize=True, strategy='peak')
    y = y_norm_t.squeeze(0).numpy()

    # Octave-shift copies
    y_1ot = librosa.effects.pitch_shift(y, sr=sr, n_steps=-12, res_type="kaiser_best")
    y_2ot = librosa.effects.pitch_shift(y, sr=sr, n_steps=-24, res_type="kaiser_best")

    # Mix
    mix = y + y_1ot + y_2ot

    # Normalize the mixed signal (RMS-based)
    rms = np.sqrt(np.mean(mix ** 2) + 1e-12)  # Add epsilon to avoid division by zero
    mix /= rms * np.sqrt(2)  # Normalize RMS to √2 for consistent loudness

    # Apply high-pass filter first to remove problematic low frequencies
    # The paper noted that <10 Hz components caused "muffled" sensations
    b_hp, a_hp = _butter_highpass(sr, cutoff_hz=10.0)
    mix = lfilter(b_hp, a_hp, mix)

    # Band-pass
    b, a = _butter_bandpass(sr, centre_hz, q)
    mix_bp = lfilter(b, a, mix)

    # Resample
    mix_bp = librosa.resample(mix_bp, orig_sr=sr, target_sr=sr_out)

    # Clip for safety
    mix_bp = np.clip(mix_bp, -1.0, 1.0)

    # Write out
    out_folder = Path(out_wav).parent
    out_folder.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, mix_bp.astype(np.float32), sr_out, subtype='PCM_16')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a haptic wav from a single audio file using frequency shifting."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input WAV file."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path for the output haptic WAV file."
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist!")
        exit(1)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process_file(
        in_wav=str(input_path),
        out_wav=str(output_path),
        centre_hz=250.0,
        q=1.0
    )
    print("Done.")
