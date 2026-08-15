"""Waveform-to-frame features matching the submitted CREPE pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchcrepe
from scipy.io import wavfile
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

from .decoder import _contiguous_regions, _fill_short_gaps


TARGET_SAMPLE_RATE = 16000
FRAME_HOP_SECONDS = 0.010
FMIN_HZ = 65.0
FMAX_HZ = 1000.0
CREPE_MODEL = "full"
CREPE_MEDIAN_FILTER = 9
PITCH_MEDIAN_FILTER = 5
SPECTRAL_FRAME_SECONDS = 0.046


def read_wav_mono(path: str | Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    sample_rate, waveform = wavfile.read(path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if np.issubdtype(waveform.dtype, np.integer):
        waveform = waveform.astype(np.float32) / np.iinfo(waveform.dtype).max
    else:
        waveform = waveform.astype(np.float32)
    if sample_rate != target_sample_rate:
        divisor = math.gcd(sample_rate, target_sample_rate)
        waveform = resample_poly(
            waveform,
            target_sample_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def hz_to_midi(frequency: np.ndarray) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=np.float64)
    output = np.full(frequency.shape, np.nan, dtype=np.float64)
    valid = frequency > 0
    output[valid] = 69.0 + 12.0 * np.log2(frequency[valid] / 440.0)
    return output


def extract_crepe_track(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> pd.DataFrame:
    hop_length = int(round(sample_rate * FRAME_HOP_SECONDS))
    audio = torch.as_tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        frequency, confidence = torchcrepe.predict(
            audio,
            sample_rate,
            hop_length,
            FMIN_HZ,
            FMAX_HZ,
            model=CREPE_MODEL,
            decoder=torchcrepe.decode.viterbi,
            return_periodicity=True,
            batch_size=batch_size,
            device=device,
        )
        frequency = torchcrepe.filter.median(frequency, CREPE_MEDIAN_FILTER)
        confidence = torchcrepe.filter.median(confidence, CREPE_MEDIAN_FILTER)
    frequency_array = frequency.squeeze(0).cpu().numpy().astype(np.float64)
    confidence_array = confidence.squeeze(0).cpu().numpy().astype(np.float64)
    del audio, frequency, confidence
    times = np.arange(len(frequency_array), dtype=np.float64) * hop_length / sample_rate
    return pd.DataFrame(
        {
            "time": times,
            "f0_hz": frequency_array,
            "confidence": confidence_array,
            "midi": hz_to_midi(frequency_array),
        }
    )


def _rms_and_flux(
    waveform: np.ndarray, sample_rate: int, number_of_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    hop_length = int(round(sample_rate * FRAME_HOP_SECONDS))
    frame_length = int(round(sample_rate * SPECTRAL_FRAME_SECONDS))
    window = np.hanning(frame_length).astype(np.float32)
    rms = np.zeros(number_of_frames, dtype=np.float64)
    flux = np.zeros(number_of_frames, dtype=np.float64)
    previous_magnitude = None
    for index in range(number_of_frames):
        start = index * hop_length
        frame = waveform[start : start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        rms[index] = float(np.sqrt(np.mean(frame * frame)))
        magnitude = np.abs(np.fft.rfft(frame * window))
        if previous_magnitude is not None:
            flux[index] = float(
                np.maximum(magnitude - previous_magnitude, 0.0).sum()
                / (previous_magnitude.sum() + 1e-8)
            )
        previous_magnitude = magnitude
    return rms, flux


def _rolling_standard_deviation(values: np.ndarray, width: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(width, center=True, min_periods=1)
        .std()
        .fillna(0.0)
        .to_numpy(np.float64)
    )


def build_frame_features(
    stem: str,
    waveform: np.ndarray,
    sample_rate: int,
    crepe_track: pd.DataFrame,
    *,
    lag_sources: list[str],
    lags: list[int],
) -> pd.DataFrame:
    times = crepe_track["time"].to_numpy(np.float64)
    raw_midi = crepe_track["midi"].to_numpy(np.float64)
    confidence = crepe_track["confidence"].to_numpy(np.float64)
    frequency = crepe_track["f0_hz"].to_numpy(np.float64)
    voiced = np.isfinite(raw_midi) & (frequency > 0)
    voiced = _fill_short_gaps(voiced, maximum_gap_frames=3)
    midi_filled = (
        pd.Series(raw_midi)
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    midi = median_filter(midi_filled, size=PITCH_MEDIAN_FILTER)
    dm = np.gradient(midi)
    ddm = np.gradient(dm)
    rms, flux = _rms_and_flux(waveform, sample_rate, len(times))
    voice_since_start = np.zeros(len(times), dtype=np.float64)
    voice_to_end = np.zeros(len(times), dtype=np.float64)
    voice_run_position = np.zeros(len(times), dtype=np.float64)
    for start, end in _contiguous_regions(voiced):
        length = max(end - start, 1)
        relative = np.arange(length, dtype=np.float64)
        voice_since_start[start:end] = relative * FRAME_HOP_SECONDS
        voice_to_end[start:end] = (length - 1 - relative) * FRAME_HOP_SECONDS
        voice_run_position[start:end] = relative / max(length - 1, 1)
    features = pd.DataFrame(
        {
            "stem": stem,
            "time": times,
            "midi": midi,
            "conf": confidence,
            "voiced": voiced.astype(np.float64),
            "dm": dm,
            "abs_dm": np.abs(dm),
            "ddm": ddm,
            "local_pitch_std": _rolling_standard_deviation(midi, 9),
            "rms": rms,
            "drms": np.gradient(rms),
            "flux": flux,
            "dflux": np.gradient(flux),
            "voice_since_start": voice_since_start,
            "voice_to_end": voice_to_end,
            "voice_run_pos": voice_run_position,
        }
    )
    for source in lag_sources:
        for lag in lags:
            features[f"{source}_lag{lag:+d}"] = features[source].shift(lag)
    return features


def extract_features_from_wav(
    path: str | Path,
    stem: str,
    *,
    lag_sources: list[str],
    lags: list[int],
    device: str = "cuda",
    batch_size: int = 32,
) -> pd.DataFrame:
    waveform, sample_rate = read_wav_mono(path)
    track = extract_crepe_track(
        waveform, sample_rate, device=device, batch_size=batch_size
    )
    return build_frame_features(
        stem,
        waveform,
        sample_rate,
        track,
        lag_sources=lag_sources,
        lags=lags,
    )
