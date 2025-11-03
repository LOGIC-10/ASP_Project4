from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from mtrf import TRF

from .data import TrialSet
from .metrics import pearsonr_np


@dataclass
class ReconstructionTrialResult:
    correlation: float
    prediction: np.ndarray


@dataclass
class ReconstructionResult:
    trials: List[ReconstructionTrialResult]

    @property
    def average_correlation(self) -> float:
        if not self.trials:
            return 0.0
        return float(np.mean([trial.correlation for trial in self.trials]))


def evaluate_reconstruction(model: TRF, trials: TrialSet) -> ReconstructionResult:
    trial_results: List[ReconstructionTrialResult] = []
    for att_trial, eeg_trial in zip(trials.wav_attended, trials.eeg):
        pred, corr = model.predict(
            stimulus=[att_trial.reshape(-1, 1)],
            response=[eeg_trial],
            average=True,
        )
        trial_results.append(
            ReconstructionTrialResult(
                correlation=float(corr),
                prediction=pred[0].reshape(-1),
            )
        )
    return ReconstructionResult(trials=trial_results)


@dataclass
class DecodingTrialResult:
    r_attended: float
    r_unattended: float
    correct: bool


@dataclass
class DecodingResult:
    trials: List[DecodingTrialResult]

    @property
    def accuracy(self) -> float:
        if not self.trials:
            return 0.0
        return float(np.mean([trial.correct for trial in self.trials]))


def evaluate_decoding(
    predictions: Iterable[np.ndarray],
    trials: TrialSet,
) -> DecodingResult:
    trial_stats: List[DecodingTrialResult] = []
    for pred_env, att_trial, unatt_trial in zip(
        predictions, trials.wav_attended, trials.wav_unattended
    ):
        r_att = pearsonr_np(pred_env, att_trial)
        r_unatt = pearsonr_np(pred_env, unatt_trial)
        trial_stats.append(
            DecodingTrialResult(
                r_attended=r_att,
                r_unattended=r_unatt,
                correct=r_att > r_unatt,
            )
        )
    return DecodingResult(trials=trial_stats)


@dataclass
class WindowedDecodingResult:
    window_length: float
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return float(self.correct / self.total) if self.total else 0.0


def evaluate_windowed_decoding(
    predictions: Iterable[np.ndarray],
    trials: TrialSet,
    fs: int,
    window_lengths: Iterable[float],
) -> Dict[float, WindowedDecodingResult]:
    results: Dict[float, WindowedDecodingResult] = {}
    for length in window_lengths:
        samples = int(round(length * fs))
        if samples <= 0:
            continue
        correct = 0
        total = 0
        for pred_env, att_trial, unatt_trial in zip(
            predictions, trials.wav_attended, trials.wav_unattended
        ):
            n = min(len(pred_env), len(att_trial), len(unatt_trial))
            if n < samples:
                continue
            for start in range(0, n - samples + 1, samples):
                end = start + samples
                seg_pred = pred_env[start:end]
                seg_att = att_trial[start:end]
                seg_unatt = unatt_trial[start:end]
                if (
                    seg_pred.std() < 1e-12
                    or seg_att.std() < 1e-12
                    or seg_unatt.std() < 1e-12
                ):
                    continue
                r_att = pearsonr_np(seg_pred, seg_att)
                r_unatt = pearsonr_np(seg_pred, seg_unatt)
                correct += int(r_att > r_unatt)
                total += 1
        results[length] = WindowedDecodingResult(
            window_length=float(length),
            correct=correct,
            total=total,
        )
    return results
