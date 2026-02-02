"""
Pitch Match audio-to-vibration translator after

@article{kim2023sound, title={Sound-to-touch crossmodal pitch matching for short sounds}, author={Kim, Dong-Geun and Lee, Jungeun and Yun, Gyeore and Tan, Hong Z and Choi, Seungmoon}, journal={IEEE Transactions on Haptics}, volume={17}, number={1}, pages={2--7}, year={2023}, publisher={IEEE} }

Modified and implemented for longer sound and clips

This python script is only used for Sound2Hap web tool's generation environment.
For the actual Pitch Match vibration dataset used in the study, please follow the MATLAB version code.

"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample_poly

# --- ISO 532-1 loudness via MOSQITO (time-varying & specific loudness) ---
# Docs: MoSQITo supports ISO 532-1 time-varying loudness and specific loudness (Bark). :contentReference[oaicite:1]{index=1}
try:
    # Time-varying loudness (Zwicker, ISO 532-1)
    from mosqito.functions.loudness_zwst._loudness_zwst import loudness_zwst  # stationary
    from mosqito.functions.loudness_zwtv._loudness_zwtv import loudness_zwtv  # time-varying
    MOSQITO_AVAILABLE = True
except Exception:
    MOSQITO_AVAILABLE = False


# ----------------------------- CONFIG --------------------------------------

@dataclass
class Config:
    # MATLAB: containers.Map([2,3,9,12,24], [-0.005, 0.003, -0.015, 0.008, 0.008])
    regressionCoeffs: dict
    vibrationFreqRange: tuple       # (minHz, maxHz)
    binSizeMs: float
    overlapRatio: float
    smoothingWindow: int            # moving average window (bins)
    inputSampleRate: int
    outputSampleRate: int


def get_config() -> Config:
    return Config(
        regressionCoeffs={2: -0.005, 3: 0.003, 9: -0.015, 12: 0.008, 24: 0.008},
        vibrationFreqRange=(50.0, 398.0),
        binSizeMs=10.0,
        overlapRatio=0.5,
        smoothingWindow=3,
        inputSampleRate=44100,
        outputSampleRate=8000,
    )


# ------------------------- UTILS / NORMALIZATION ---------------------------

def normalize_audio(audio: np.ndarray, do_normalize: bool) -> np.ndarray:
    # MATLAB normalizeAudio()
    scale_peak = 10 ** (-1 / 20)  # -1 dBFS
    normalize_peak = 1.0
    wav_max = np.max(np.abs(audio)) + 1e-12
    rescaling = min(max(1.0, normalize_peak / wav_max), scale_peak / wav_max)
    if do_normalize or (rescaling < 1.0):
        audio = audio * rescaling
    return audio


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


# -------------------- ISO 532-1 SPECIFIC & TOTAL LOUDNESS ------------------

def _specific_and_total_loudness_bark(audio_bin: np.ndarray, sr: int):
    """
    Returns (specific_loudness_24, total_loudness_scalar)

    - specific_loudness_24: 24 Bark-band values (sone/Bark), aggregated from the
      ISO 532-1 240-bin (0.1 Bark steps) specific loudness.
    - total_loudness_scalar: scalar loudness (sones).
    """
    if not MOSQITO_AVAILABLE:
        # Fallback: approximate with envelope energy (kept to preserve pipeline)
        # This keeps everything running if mosqito isn't installed.
        env = np.abs(audio_bin)
        total = float(np.mean(env))
        spec24 = np.zeros(24, dtype=np.float32)
        spec24[0] = total
        return spec24, total

    # mosqito loudness_zwtv expects a waveform at (ideally) 48k, but handles others.
    # It returns:
    #   - N: loudness vs. time (sones)
    #   - N_spec: specific loudness vs. time in Bark bands with 0.1 step (shape [T, 240])
    # We'll compute for a short window (our bin) and then reduce over time.
    # Note: If a bin is very short, zwtv can still return arrays (could be length 1).
    try:
        results = loudness_zwtv(audio_bin, sr, field_type="free")  # 'free' field to match MATLAB. :contentReference[oaicite:2]{index=2}
        # results keys may include 'N' and 'N_specific'
        N_time = np.asarray(results["N"]).reshape(-1)            # [T]
        N_spec = np.asarray(results["N_specific"])               # [T, 240]
        # Collapse time within the bin:
        total_loudness = float(np.mean(N_time)) if N_time.size else 0.0

        # Aggregate 240 → 24 bands by summing each contiguous group of 10 bins.
        if N_spec.ndim == 2 and N_spec.shape[1] >= 240:
            spec_time_mean = np.mean(N_spec, axis=0)  # [240]
            spec24 = np.zeros(24, dtype=np.float32)
            for i in range(24):
                start = i * 10
                end = start + 10
                spec24[i] = float(np.sum(spec_time_mean[start:end]))
        else:
            # Unexpected shape: fall back to interpolation to 24
            if N_spec.ndim == 1:  # [240] or other
                vec = N_spec
            else:
                vec = np.mean(N_spec, axis=0) if N_spec.size else np.zeros(240)
            idx = np.linspace(0, len(vec) - 1, 24)
            spec24 = np.interp(idx, np.arange(len(vec)), vec).astype(np.float32)

        # sanitize
        spec24[~np.isfinite(spec24)] = 0.0
        return spec24, total_loudness
    except Exception:
        # Robust fallback
        env = np.abs(audio_bin)
        total = float(np.mean(env))
        spec24 = np.zeros(24, dtype=np.float32)
        spec24[0] = total
        return spec24, total


# -------------------------- CORE MODEL COMPONENTS --------------------------

def predict_vibration_frequency(specific_loudness_24: np.ndarray, cfg: Config) -> float:
    """
    Linear regression over select Bark bands -> predicted vib frequency (Hz),
    then clamped to cfg.vibrationFreqRange.
    Mirrors MATLAB predictVibrationFrequency()
    """
    predicted = 0.0
    for bark_band, coeff in cfg.regressionCoeffs.items():
        idx = int(bark_band) - 1  # MATLAB 1-based -> Python 0-based
        if 0 <= idx < len(specific_loudness_24):
            predicted += coeff * float(specific_loudness_24[idx]) * 1000.0

    predicted = abs(predicted)
    vmin, vmax = cfg.vibrationFreqRange
    return float(np.clip(predicted, vmin, vmax))


def analyze_audio_bins(audio: np.ndarray, sr: int, cfg: Config):
    """
    Break long audio into overlapping bins; for each bin:
      - compute specific loudness (24 Bark) & total loudness
      - predict vib frequency from specific loudness
      - store (time_center, freq, amplitude)
    Apply small moving average (optional) on the frequency track.
    """
    bin_size = int(round(cfg.binSizeMs * sr / 1000.0))
    hop = max(1, int(round(bin_size * (1.0 - cfg.overlapRatio))))
    if bin_size < 2:
        bin_size = 2
    starts = np.arange(0, max(1, len(audio) - bin_size + 1), hop, dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)

    times = (starts + bin_size / 2.0) / float(sr)
    freqs = np.zeros(starts.size, dtype=np.float32)
    amps = np.zeros(starts.size, dtype=np.float32)
    win = get_window("hann", bin_size, fftbins=False).astype(np.float32)

    for i, s in enumerate(starts):
        e = min(s + bin_size, len(audio))
        chunk = np.zeros(bin_size, dtype=np.float32)
        seg = audio[s:e]
        chunk[: len(seg)] = seg
        chunk *= win

        if rms(chunk) < 1e-3:
            freqs[i] = freqs[i - 1] if i > 0 else np.mean(cfg.vibrationFreqRange)
            amps[i] = 0.0
            continue

        spec24, loud = _specific_and_total_loudness_bark(chunk, sr)
        freqs[i] = predict_vibration_frequency(spec24, cfg)
        amps[i] = float(loud)

    # smoothing on frequency
    if cfg.smoothingWindow > 1 and len(freqs) > cfg.smoothingWindow:
        k = cfg.smoothingWindow
        kernel = np.ones(k, dtype=np.float32) / k
        freqs = np.convolve(freqs, kernel, mode="same")

    return times.astype(np.float64), freqs.astype(np.float64), amps.astype(np.float64)


def generate_time_varying_vibration(audio: np.ndarray, sr: int, cfg: Config):
    bin_t, bin_f, bin_a = analyze_audio_bins(audio, sr, cfg)

    t = np.arange(len(audio), dtype=np.float64) / float(sr)

    # Interpolate frequency & amplitude from bin grid to full rate
    if len(bin_t) == 1:
        f_inst = np.full_like(t, bin_f[0], dtype=np.float64)
        a_inst = np.full_like(t, bin_a[0], dtype=np.float64)
    else:
        # Frequency: PCHIP if available, otherwise linear
        try:
            from scipy.interpolate import PchipInterpolator
            f_inst = PchipInterpolator(bin_t, bin_f, extrapolate=True)(t)
        except Exception:
            f_inst = np.interp(t, bin_t, bin_f, left=bin_f[0], right=bin_f[-1])
        a_inst = np.interp(t, bin_t, bin_a, left=bin_a[0], right=bin_a[-1])

    # RMS-based amplitude normalization (bounded)
    rms_current = rms(a_inst)
    rms_norm = np.sqrt(2.0)
    rms_amplify = max(1.0, min(1.2, 1.0 / (rms_current * rms_norm))) if rms_current > 0 else 1.0
    a_inst = a_inst * (rms_norm * rms_amplify) if rms_current > 0 else np.full_like(t, 0.1)

    # Phase accumulation: phi[n] = phi[n-1] + 2π f[n-1] dt
    dt = 1.0 / float(sr)
    phi = np.empty_like(t)
    phi[0] = 0.0
    # cumulative sum of frequency
    phi[1:] = 2.0 * np.pi * np.cumsum(f_inst[:-1]) * dt

    v = a_inst * np.sin(phi)

    # 10 ms fade in/out
    fade_len = int(round(0.01 * sr))
    if len(v) > 2 * fade_len and fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        v[:fade_len] *= fade_in
        v[-fade_len:] *= fade_out

    return v.astype(np.float32), f_inst.astype(np.float32), a_inst.astype(np.float32)


def generate_vibration_signal(audio: np.ndarray, sr: int, cfg: Config):
    v, f_arr, a_arr = generate_time_varying_vibration(audio, sr, cfg)
    analysis_info = {
        "method": "time_varying",
        "duration": len(audio) / float(sr),
        "freqMean": float(np.mean(f_arr)),
        "freqRange": (float(np.min(f_arr)), float(np.max(f_arr))),
        "freqStd": float(np.std(f_arr)),
    }
    return v, f_arr, a_arr, analysis_info


# --------------------------------- I/O -------------------------------------

def _read_mono(path: str):
    x, sr = sf.read(path, always_2d=False)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, sr


def _write_int16_wav(path: str, y: np.ndarray, sr: int):
    # peak normalize to [-1,1], then write 16-bit PCM
    y = y / (np.max(np.abs(y)) + 1e-12)
    sf.write(path, y, sr, subtype="PCM_16")


def process_audio_file(input_file: str, output_file: str, cfg: Config):
    print(f"Processing: {input_file}")
    audio, sr = _read_mono(input_file)
    duration = len(audio) / float(sr)

    audio = normalize_audio(audio, True)

    v, f_arr, a_arr, info = generate_vibration_signal(audio, sr, cfg)

    # Resample to target (8 kHz) like MATLAB
    fs_out = cfg.outputSampleRate
    if sr != fs_out:
        from fractions import Fraction
        frac = Fraction(fs_out, sr).limit_denominator(1000)
        v = resample_poly(v, frac.numerator, frac.denominator)

    out_dir = Path(output_file).parent
    if str(out_dir) and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {out_dir}")

    _write_int16_wav(output_file, v, fs_out)

    result = {
        "inputFile": str(input_file),
        "outputFile": str(output_file),
        "duration": duration,
        "originalSr": sr,
        "targetSr": fs_out,
        "analysisInfo": info,
    }
    return True, result

# ------------------------------ MAIN -----------------------------
if __name__ == "__main__":
    inputFilePath = "test.wav"
    outputFilePath = "test-pitch.wav"

    print("Starting single file processing...")
    cfg = get_config()

    try:
        ok, res = process_audio_file(inputFilePath, outputFilePath, cfg)
        if ok:
            print("\nProcessing complete.")
            print(f"Input:  {res['inputFile']}")
            print(f"Output: {res['outputFile']}")
        else:
            print(f"\nProcessing failed for: {inputFilePath}")
    except Exception as e:
        print("\nAn unexpected error occurred during processing:")
        print(str(e))