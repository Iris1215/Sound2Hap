import torch
import torchaudio
import typing as tp


def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 14,
                       loudness_compressor: bool = False, energy_floor: float = 2e-3):
    """Normalize an input signal to a user loudness in dB LKFS.
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.

    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        loudness_compressor (bool): Uses tanh for soft clipping.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        torch.Tensor: Loudness normalized output data.
    """
    energy = wav.pow(2).mean().sqrt().item()
    if energy < energy_floor:
        return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    input_loudness_db = transform(wav).item()
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    if loudness_compressor:
        output = torch.tanh(output)
    assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
    return output


def _clip_wav(wav: torch.Tensor, log_clipping: bool = False, stem_name: tp.Optional[str] = None) -> None:
    """Utility function to clip the audio with logging if specified."""
    max_scale = wav.abs().max()
    if log_clipping and max_scale > 1:
        clamp_prob = (wav.abs() > 1).float().mean().item()
        print(f"CLIPPING {stem_name or ''} happening with proba (a bit of clipping is okay):",
              clamp_prob, "maximum scale: ", max_scale.item(), file=sys.stderr)
    wav.clamp_(-1, 1)


def normalize_audio(wav: torch.Tensor, normalize: bool = True,
                    strategy: str = 'peak', peak_clip_headroom_db: float = 1, peak_normalize_db_clamp: float = 0,
                    rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                    loudness_compressor: bool = False, log_clipping: bool = False,
                    sample_rate: tp.Optional[int] = None,
                    stem_name: tp.Optional[str] = None) -> torch.Tensor:
    """Normalize the audio according to the prescribed strategy (see after).

    Args:
        wav (torch.Tensor): Audio data.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        peak_normalize_db_clamp (float): Maximum normalization in dB when doing 'peak' strategy (e.g. set to -20 dB for max amplify to 0.1).
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): If True, uses tanh based soft clipping.
        log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        sample_rate (int): Sample rate for the audio data (required for loudness).
        stem_name (str, optional): Stem name for clipping logging.
    Returns:
        torch.Tensor: Normalized audio.
    """
    scale_peak = 10 ** (-peak_clip_headroom_db / 20)
    normalize_peak = 10 ** (peak_normalize_db_clamp / 20)
    scale_rms = 10 ** (-rms_headroom_db / 20)
    if strategy == 'peak':
        wav_max = wav.abs().max()
        rescaling = (scale_peak / wav_max).clamp(max=(normalize_peak / wav_max).clamp(min=1))
        if normalize or rescaling < 1:
            wav = wav * rescaling
    elif strategy == 'clip':
        wav = wav.clamp(-scale_peak, scale_peak)
    elif strategy == 'rms':
        mono = wav.mean(dim=0)
        rescaling = scale_rms / mono.pow(2).mean().sqrt()
        if normalize or rescaling < 1:
            wav = wav * rescaling
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    elif strategy == 'loudness':
        assert sample_rate is not None, "Loudness normalization requires sample rate."
        wav = normalize_loudness(wav, sample_rate, loudness_headroom_db, loudness_compressor)
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    else:
        assert wav.abs().max() < 1
        assert strategy == '' or strategy == 'none', f"Unexpected strategy: '{strategy}'"
    return wav
