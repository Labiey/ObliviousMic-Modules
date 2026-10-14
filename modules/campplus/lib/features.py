# SPDX-License-Identifier: Apache-2.0
# Adapted for module-local distribution in 2026; see NOTICE and provenance.md.
"""Frozen CAM++ fbank functions, extracted without model/download management."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SPEAKER_ENCODER_SAMPLE_RATE = 16000
SPEAKER_ENCODER_FEATURE_DIMENSION = 80


class SpeakerEncoderError(RuntimeError):
    pass


def compute_kaldi_fbank(
    samples: NDArray[np.float32],
    sample_rate: int = SPEAKER_ENCODER_SAMPLE_RATE,
) -> NDArray[np.float32]:
    """Compute Kaldi-compatible 80-bin log-mel features for a 3D-Speaker model."""

    audio = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
    if sample_rate != SPEAKER_ENCODER_SAMPLE_RATE:
        raise SpeakerEncoderError(
            f"fbank requires {SPEAKER_ENCODER_SAMPLE_RATE} Hz audio; got {sample_rate} Hz"
        )
    frame_length = round(sample_rate * 0.025)
    frame_shift = round(sample_rate * 0.010)
    if audio.shape[0] < frame_length:
        raise SpeakerEncoderError("audio is too short to compute one fbank frame")

    # sherpa-onnx uses Kaldi's snip_edges=false policy for speaker models:
    # frame zero is centred at half a frame shift and edge samples are mirrored.
    frame_count = (audio.shape[0] + frame_shift // 2) // frame_shift
    first_start = frame_shift // 2 - frame_length // 2
    last_end = (frame_count - 1) * frame_shift + first_start + frame_length
    left_padding = max(0, -first_start)
    right_padding = max(0, last_end - audio.shape[0])
    padded = np.pad(audio, (left_padding, right_padding), mode="symmetric")
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[
        : frame_count * frame_shift : frame_shift
    ].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= np.float32(0.97) * frames[:, :-1]
    frames[:, 0] *= np.float32(1.0 - 0.97)
    window = _povey_window(frame_length)
    frames *= window[np.newaxis, :]

    fft_size = 1 << (frame_length - 1).bit_length()
    spectrum = np.fft.rfft(frames, n=fft_size, axis=1)
    # Kaldi's mel-bank implementation intentionally excludes the Nyquist bin.
    power = np.asarray(spectrum.real**2 + spectrum.imag**2, dtype=np.float32)[:, : fft_size // 2]
    mel_banks = _mel_filter_banks(sample_rate, fft_size, SPEAKER_ENCODER_FEATURE_DIMENSION)
    energies = power @ mel_banks.T
    np.maximum(energies, np.finfo(np.float32).eps, out=energies)
    np.log(energies, out=energies)
    return np.ascontiguousarray(energies, dtype=np.float32)


def _povey_window(length: int) -> NDArray[np.float32]:
    indices = np.arange(length, dtype=np.float64)
    base = 0.5 - 0.5 * np.cos(2.0 * np.pi * indices / (length - 1))
    return np.asarray(base**0.85, dtype=np.float32)


def _mel_filter_banks(sample_rate: int, fft_size: int, bins: int) -> NDArray[np.float32]:
    fft_frequencies = np.arange(fft_size // 2, dtype=np.float32) * np.float32(
        sample_rate / fft_size
    )
    low_mel = _mel_scale(np.float32(20.0))
    high_mel = _mel_scale(np.float32(sample_rate / 2 - 400.0))
    mel_delta = np.float32((high_mel - low_mel) / (bins + 1))
    mel_points = low_mel + np.arange(bins + 2, dtype=np.float32) * mel_delta
    fft_mels = _mel_scale(fft_frequencies)
    banks = np.zeros((bins, fft_frequencies.shape[0]), dtype=np.float32)
    for index in range(bins):
        left, center, right = mel_points[index : index + 3]
        lower = (fft_mels - left) / (center - left)
        upper = (right - fft_mels) / (right - center)
        banks[index] = np.maximum(0.0, np.minimum(lower, upper))
    return banks


def _mel_scale(
    frequency: np.float32 | NDArray[np.float32],
) -> np.float32 | NDArray[np.float32]:
    return np.float32(1127.0) * np.log1p(np.asarray(frequency, dtype=np.float32) / 700.0)
