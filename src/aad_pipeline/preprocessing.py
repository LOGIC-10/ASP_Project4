from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .data import TrialSet


@dataclass
class NormalizationStats:
    eeg_mean: np.ndarray
    eeg_std: np.ndarray
    wav_att_mean: float
    wav_att_std: float
    wav_unatt_mean: float
    wav_unatt_std: float


def _mean_std(trials: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    concatenated = np.concatenate([trial for trial in trials], axis=0)
    mean = concatenated.mean(axis=0)
    std = concatenated.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def compute_normalization(trials: TrialSet) -> NormalizationStats:
    eeg_mean, eeg_std = _mean_std([trial.astype(np.float64) for trial in trials.eeg])
    wav_att_mean, wav_att_std = _mean_std([trial.reshape(-1, 1) for trial in trials.wav_attended])
    wav_unatt_mean, wav_unatt_std = _mean_std([trial.reshape(-1, 1) for trial in trials.wav_unattended])
    return NormalizationStats(
        eeg_mean=eeg_mean,
        eeg_std=eeg_std,
        wav_att_mean=float(wav_att_mean),
        wav_att_std=float(wav_att_std),
        wav_unatt_mean=float(wav_unatt_mean),
        wav_unatt_std=float(wav_unatt_std),
    )


def apply_normalization(trials: TrialSet, stats: NormalizationStats) -> TrialSet:
    eeg = [((trial - stats.eeg_mean) / stats.eeg_std).astype(np.float64) for trial in trials.eeg]
    wav_att = [((trial - stats.wav_att_mean) / stats.wav_att_std).astype(np.float64) for trial in trials.wav_attended]
    wav_unatt = [((trial - stats.wav_unatt_mean) / stats.wav_unatt_std).astype(np.float64) for trial in trials.wav_unattended]
    return TrialSet(eeg=eeg, wav_attended=wav_att, wav_unattended=wav_unatt)
