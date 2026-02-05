# This script implements the frequency shifting algorithm described in the paper:
# @inproceedings{sung2025hapticgen, title={HapticGen: Generative Text-to-Vibration Model for Streamlining Haptic Design}, author={Sung, Youjin and John, Kevin and Yoon, Sang Ho and Seifi, Hasti}, booktitle={Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems}, pages={1--24}, year={2025}

import numpy as np
import math
import os
import soundfile as sf
import argparse
import torch
import argparse
from pathlib import Path
from utils.normalization import normalize_audio


WANTED_BIN_SIZE_SEC: float = 0.010 # 10ms, window length
BASE_FREQ: float = 200.0 # 200Hz, center pitch
def amp_env_on_wav_norm(wav_norm: np.ndarray, input_sample_rate: int, output_sample_rate: int): # wav_norm is a mono audio clip already normalized to [-1.0, 1.0]
    wav_norm = wav_norm.squeeze() # (N,1) to (N,)

    num_samples = len(wav_norm) # number of samples in the input audio
    duration_sec = num_samples / input_sample_rate # duration in seconds of the input audio
    samples_per_bin = int(WANTED_BIN_SIZE_SEC * input_sample_rate) # number of samples per bin based on the desired window length
    num_bins = num_samples // samples_per_bin # number of bins we can create from the input audio
    # print(f"wav_norm.shape: {wav_norm.shape}, samples_per_bin: {samples_per_bin}, num_bins: {num_bins}")

    wav_chunks = np.array_split(wav_norm, num_bins) # split the audio into chunks of equal size
    rms_bins = np.array([np.sqrt(np.mean(chunk ** 2)) for chunk in wav_chunks]) # calculate the RMS of each chunk
    #assert num_bins == len(rms_bins), f"num_bins: {num_bins}, len(rms_bins): {len(rms_bins)}"
    rms_max = np.max(rms_bins)  # find the maximum RMS value across all bins
    rms_norm = np.sqrt(2)  # normalize RMS to a value that will give us a peak amplitude of 1.0 when multiplied by the sine wave
    rms_amplify = max(1.0, min(1.2, 1.0 / (rms_max * rms_norm))) # ensure we amplify the RMS to avoid clipping, but not too much
    rms_norm_amp = rms_norm * rms_amplify
    out_samples = int(duration_sec * output_sample_rate) # number of samples in the output audio
    # print(f"duration_sec: {duration_sec}, out_samples: {out_samples}, rms_max: {rms_max}, rms_norm_amp: {rms_norm_amp}")

    phase_acc = 0.0 # phase accumulator for the sine wave
    output = np.zeros(out_samples) # output array to hold the generated audio samples
    for i in range(out_samples):
        t = i / output_sample_rate # current time in seconds for the output sample
        t_prog = t / duration_sec # progress through the duration of the input audio

        bin_fi = t_prog * num_bins # fractional bin index based on the progress through the input audio
        bin_lo = int(bin_fi) # lower bin index
        bin_hi = min(num_bins - 1, int(math.ceil(bin_fi))) # upper bin index, ensuring we don't go out of bounds
        bin_fr = bin_fi - bin_lo # fractional part of the bin index
        rms = (rms_bins[bin_lo] * (1.0 - bin_fr) + rms_bins[bin_hi] * bin_fr) * rms_norm_amp # interpolate RMS between the two bins
        # freq = 0.0 # no freq bins
        freq_offset = (rms - 0.3) * 100.0 # Map the instantaneous RMS to a small frequency offset: When rms = 0.3 → offset 0 Hz. Each +0.01 RMS raises the pitch by +1 Hz.

        phase_delta = 2.0 * math.pi * (BASE_FREQ + freq_offset) / output_sample_rate # Numerically controlled oscillator (Direct digital synthesis), so can change freq without phase discontinuity
        phase_acc = (phase_acc + phase_delta) % (2.0 * math.pi) # Keep phase_acc in the range 0, 2π

        sample = rms * math.sin(phase_acc) # Generate the sine wave sample at the current phase

        output[i] = sample

    return output

def process_file(input_path: str, output_path: str):
    output_sample_rate = 8000
    wav_data, sr = sf.read(input_path)
    wav_tensor = torch.from_numpy(wav_data).float().unsqueeze(0)

    wav_norm_tensor = normalize_audio(
        wav_tensor,
        normalize=True,
        strategy="peak",
        peak_clip_headroom_db=0,
        peak_normalize_db_clamp=0
    )

    wav = wav_norm_tensor.squeeze(0).numpy()
    env_signal = amp_env_on_wav_norm(wav, sr, output_sample_rate)

    sf.write(output_path, env_signal, output_sample_rate, subtype='PCM_16')
    # sf.write(output_path, env_signal, output_sample_rate, subtype='PCM_U8')
    print(f"Processed: '{os.path.basename(input_path)}' -> '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a haptic wav from a single audio file."
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

    # Ensure output directory exists
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process_file(str(input_path), str(output_path))
    
    print("Done.")
