from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(SRC))

from aad_pipeline import (
    apply_normalization,
    compute_normalization,
    evaluate_decoding,
    evaluate_reconstruction,
    load_subject_data,
)
from aad_pipeline.data import list_subjects
from aad_pipeline.model import train_final_model


TMIN = -0.1
TMAX = 0.4
SUMMARY_PATH = RESULTS / "subject_summary.csv"
WINDOW_PATH = RESULTS / "windowed_accuracy.csv"
CV_PATTERN = "{subject}_cv.json"

def load_best_lambdas() -> Dict[str, float]:
    best: Dict[str, float] = {}
    with SUMMARY_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            best[row["subject"]] = float(row["best_lambda"])
    return best


def make_lambda_curves(subjects: List[str]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax_corr, ax_acc = axes
    for subject in subjects:
        path = RESULTS / CV_PATTERN.format(subject=subject)
        with path.open() as f:
            records = json.load(f)
        lambdas = np.array([rec["lambda"] for rec in records])
        corr = np.array([rec["reconstruction_score"] for rec in records])
        acc = np.array([rec["decoding_accuracy"] for rec in records])
        order = np.argsort(lambdas)
        lambdas, corr, acc = lambdas[order], corr[order], acc[order]
        ax_corr.plot(lambdas, corr, marker="o", label=subject)
        ax_acc.plot(lambdas, acc, marker="o", label=subject)
    ax_corr.set_xscale("log")
    ax_acc.set_xscale("log")
    ax_corr.set_ylabel("Mean Pearson r")
    ax_acc.set_ylabel("Decoding accuracy")
    ax_acc.set_xlabel("Regularization $\\lambda$")
    ax_corr.set_title("Cross-validated reconstruction across $\\lambda$")
    ax_acc.set_title("Cross-validated decoding accuracy across $\\lambda$")
    ax_corr.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_acc.grid(True, which="both", linestyle="--", alpha=0.4)
    handles, labels = ax_corr.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=8)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS / "lambda_cv_curves.png", dpi=300)
    plt.close(fig)


def gather_trial_metrics(subjects: List[str], lambdas: Dict[str, float]):
    trial_corr: Dict[str, List[float]] = defaultdict(list)
    trial_accuracy: Dict[str, List[bool]] = defaultdict(list)
    margin_values: List[float] = []
    error_signal: Tuple[np.ndarray, float] | None = None

    for subject in subjects:
        subject_path = ROOT / "Project4_Data" / subject
        data = load_subject_data(subject_path)
        stats = compute_normalization(data.train)
        test_norm = apply_normalization(data.test, stats)
        model = train_final_model(
            train_trials=data.train,
            stats=stats,
            lambda_value=lambdas[subject],
            fs=data.fs,
            tmin=TMIN,
            tmax=TMAX,
        )
        test_recon = evaluate_reconstruction(model, test_norm)
        predictions = [trial.prediction for trial in test_recon.trials]
        decoding = evaluate_decoding(predictions, test_norm)
        for trial_idx, (trial, pred) in enumerate(zip(test_recon.trials, predictions)):
            trial_corr[subject].append(trial.correlation)
            trial_accuracy[subject].append(decoding.trials[trial_idx].correct)
            margin = (
                decoding.trials[trial_idx].r_attended
                - decoding.trials[trial_idx].r_unattended
            )
            margin_values.append(margin)
        # keep first subject/trial reconstruction error for PSD
        if subject == "S1" and error_signal is None:
            err = predictions[0] - test_norm.wav_attended[0]
            error_signal = (err, data.fs)

    return trial_corr, trial_accuracy, margin_values, error_signal


def make_subject_boxplots(trial_corr, trial_accuracy):
    subjects = sorted(trial_corr.keys(), key=lambda s: int(s[1:]))
    corr_data = [trial_corr[s] for s in subjects]
    acc_means = [np.mean(trial_accuracy[s]) for s in subjects]

    fig, (ax_corr, ax_acc) = plt.subplots(2, 1, figsize=(8, 8))
    ax_corr.boxplot(corr_data, tick_labels=subjects, showmeans=True)
    ax_corr.set_ylabel("Test Pearson r")
    ax_corr.set_title("Per-subject distribution of test reconstruction scores")
    ax_corr.grid(axis="y", linestyle="--", alpha=0.4)

    ax_acc.bar(subjects, acc_means, color="#4c72b0")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_ylabel("Test accuracy")
    ax_acc.set_title("Per-subject attention decoding accuracy")
    ax_acc.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(RESULTS / "subject_performance_boxplots.png", dpi=300)
    plt.close(fig)


def make_error_spectrum(error_signal: Tuple[np.ndarray, float] | None) -> None:
    if error_signal is None:
        return
    error, fs = error_signal
    freqs, psd = welch(error, fs=fs, nperseg=min(1024, len(error)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(freqs, psd)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")
    ax.set_title("Spectrum of reconstruction error (S1 test trial 1)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(RESULTS / "reconstruction_error_spectrum.png", dpi=300)
    plt.close(fig)


def make_margin_histogram(margins: List[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(margins, bins=30, color="#55a868", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Correlation margin ($r_{att} - r_{unatt}$)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of decoding correlation margins (test trials)")
    fig.tight_layout()
    fig.savefig(RESULTS / "decoding_margin_hist.png", dpi=300)
    plt.close(fig)


def make_window_trajectories() -> None:
    records = defaultdict(dict)
    with WINDOW_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row["subject"]
            length = float(row["window_length"])
            accuracy = float(row["accuracy"])
            records[subject][length] = accuracy
    lengths = sorted({length for data in records.values() for length in data})
    subjects = sorted(records.keys(), key=lambda s: int(s[1:]))

    fig, ax = plt.subplots(figsize=(7, 5))
    for subject in subjects:
        accs = [records[subject].get(length, np.nan) for length in lengths]
        ax.plot(lengths, accs, marker="o", label=subject)
    ax.set_xlabel("Window length (s)")
    ax.set_ylabel("Decoding accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Window-length trajectories by subject")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncol=4, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "window_accuracy_subjects.png", dpi=300)
    plt.close(fig)


def main() -> None:
    if not SUMMARY_PATH.exists() or not WINDOW_PATH.exists():
        raise FileNotFoundError("Run run_pipeline.py before generating figures.")
    subjects = [p.name for p in list_subjects(ROOT / "Project4_Data")]
    lambdas = load_best_lambdas()
    make_lambda_curves(subjects)
    trial_corr, trial_acc, margins, error_signal = gather_trial_metrics(subjects, lambdas)
    make_subject_boxplots(trial_corr, trial_acc)
    make_error_spectrum(error_signal)
    make_margin_histogram(margins)
    make_window_trajectories()


if __name__ == "__main__":
    main()
