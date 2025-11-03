from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from mtrf import TRF
from sklearn.model_selection import KFold

from .data import TrialSet
from .metrics import pearsonr_np
from .preprocessing import NormalizationStats, apply_normalization, compute_normalization


@dataclass
class LambdaEvaluation:
    lambda_value: float
    reconstruction_score: float
    decoding_accuracy: float


@dataclass
class CrossValidationResult:
    evaluations: List[LambdaEvaluation]
    best_lambda: float


def _subset(trials: TrialSet, indices: Sequence[int]) -> TrialSet:
    return TrialSet(
        eeg=[trials.eeg[i] for i in indices],
        wav_attended=[trials.wav_attended[i] for i in indices],
        wav_unattended=[trials.wav_unattended[i] for i in indices],
    )


def _reshape_envelope(trials: Iterable[np.ndarray]) -> List[np.ndarray]:
    return [trial.reshape(-1, 1) for trial in trials]


def cross_validate_lambda(
    train_trials: TrialSet,
    lambda_grid: Sequence[float],
    fs: int,
    tmin: float,
    tmax: float,
    k_folds: int = 5,
    random_state: int = 42,
) -> CrossValidationResult:
    """Perform cross-validation to select the regularization parameter."""

    n_trials = len(train_trials.eeg)
    if n_trials < k_folds:
        raise ValueError("Number of folds cannot exceed number of trials")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    evaluations: List[LambdaEvaluation] = []

    for lam in lambda_grid:
        fold_recon: List[float] = []
        fold_acc: List[float] = []
        for train_idx, val_idx in kf.split(np.arange(n_trials)):
            train_subset = _subset(train_trials, train_idx)
            val_subset = _subset(train_trials, val_idx)
            stats = compute_normalization(train_subset)
            train_norm = apply_normalization(train_subset, stats)
            val_norm = apply_normalization(val_subset, stats)

            model = TRF(direction=-1, preload=True)
            model.train(
                stimulus=_reshape_envelope(train_norm.wav_attended),
                response=train_norm.eeg,
                fs=fs,
                tmin=tmin,
                tmax=tmax,
                regularization=float(lam),
                k=0,
                verbose=False,
            )

            for att_trial, eeg_trial, unatt_trial in zip(
                val_norm.wav_attended, val_norm.eeg, val_norm.wav_unattended
            ):
                pred, corr = model.predict(
                    stimulus=[att_trial.reshape(-1, 1)],
                    response=[eeg_trial],
                    average=True,
                )
                fold_recon.append(float(corr))
                pred_env = pred[0].reshape(-1)
                r_att = pearsonr_np(pred_env, att_trial.reshape(-1))
                r_unatt = pearsonr_np(pred_env, unatt_trial.reshape(-1))
                fold_acc.append(1.0 if r_att > r_unatt else 0.0)

        evaluations.append(
            LambdaEvaluation(
                lambda_value=float(lam),
                reconstruction_score=float(np.mean(fold_recon)),
                decoding_accuracy=float(np.mean(fold_acc)),
            )
        )

    best = max(
        evaluations,
        key=lambda res: (
            res.decoding_accuracy,
            res.reconstruction_score,
            -res.lambda_value,
        ),
    )
    return CrossValidationResult(evaluations=evaluations, best_lambda=best.lambda_value)


def train_final_model(
    train_trials: TrialSet,
    stats: NormalizationStats,
    lambda_value: float,
    fs: int,
    tmin: float,
    tmax: float,
) -> TRF:
    """Train a TRF decoder on normalized training data."""

    normalized = apply_normalization(train_trials, stats)
    model = TRF(direction=-1, preload=True)
    model.train(
        stimulus=_reshape_envelope(normalized.wav_attended),
        response=normalized.eeg,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        regularization=float(lambda_value),
        k=0,
        verbose=False,
    )
    return model
