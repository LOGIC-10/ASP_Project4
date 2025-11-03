from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib.pyplot as plt
import numpy as np

from aad_pipeline import (
    CrossValidationResult,
    apply_normalization,
    compute_normalization,
    cross_validate_lambda,
    evaluate_decoding,
    evaluate_reconstruction,
    evaluate_windowed_decoding,
    load_subject_data,
)
from aad_pipeline.data import list_subjects
from aad_pipeline.model import train_final_model


DATA_ROOT = Path("Project4_Data")
RESULTS_DIR = Path("results")
TMIN = -0.1
TMAX = 0.4
LAMBDA_GRID = np.logspace(-3, 3, 13)
WINDOW_LENGTHS = [1, 5, 10, 15, 30]
K_FOLDS = 5
RANDOM_STATE = 42
EXAMPLE_SUBJECT = "S1"
EXAMPLE_TRIAL = 0


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)


def save_csv(path: Path, records: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def plot_reconstruction_example(subject_id: str, fs: int, ground_truth: np.ndarray, prediction: np.ndarray) -> None:
    time = np.arange(len(ground_truth)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(time, ground_truth, label="Attended envelope (z-scored)")
    plt.plot(time, prediction, label="Reconstruction", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title(f"Envelope reconstruction example — {subject_id}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "reconstruction_example.png", dpi=300)
    plt.close()


def plot_window_accuracy(window_records: List[Dict[str, object]]) -> None:
    lengths = sorted({record["window_length"] for record in window_records})
    mean_acc = []
    std_acc = []
    for length in lengths:
        acc = [record["accuracy"] for record in window_records if record["window_length"] == length]
        mean_acc.append(np.mean(acc))
        std_acc.append(np.std(acc))
    plt.figure(figsize=(6, 4))
    plt.errorbar(lengths, mean_acc, yerr=std_acc, fmt="-o", capsize=4)
    plt.xlabel("Window length (s)")
    plt.ylabel("Decoding accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Attention decoding vs. window length")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "aad_window_accuracy.png", dpi=300)
    plt.close()


def save_crossval_results(subject_id: str, cv_result: CrossValidationResult) -> None:
    records = [
        {
            "subject": subject_id,
            "lambda": eval.lambda_value,
            "reconstruction_score": eval.reconstruction_score,
            "decoding_accuracy": eval.decoding_accuracy,
        }
        for eval in cv_result.evaluations
    ]
    path = RESULTS_DIR / f"{subject_id}_cv.json"
    with path.open("w") as f:
        json.dump(records, f, indent=2)


def main() -> None:
    np.random.seed(RANDOM_STATE)
    ensure_results_dir()

    subject_paths = list_subjects(DATA_ROOT)
    summary_records: List[Dict[str, object]] = []
    window_records: List[Dict[str, object]] = []
    example_ready = False

    for subject_path in subject_paths:
        subject = load_subject_data(subject_path)
        cv_result = cross_validate_lambda(
            train_trials=subject.train,
            lambda_grid=LAMBDA_GRID,
            fs=subject.fs,
            tmin=TMIN,
            tmax=TMAX,
            k_folds=K_FOLDS,
            random_state=RANDOM_STATE,
        )
        save_crossval_results(subject.subject_id, cv_result)

        stats = compute_normalization(subject.train)
        train_norm = apply_normalization(subject.train, stats)
        test_norm = apply_normalization(subject.test, stats)

        model = train_final_model(
            train_trials=subject.train,
            stats=stats,
            lambda_value=cv_result.best_lambda,
            fs=subject.fs,
            tmin=TMIN,
            tmax=TMAX,
        )

        train_recon = evaluate_reconstruction(model, train_norm)
        test_recon = evaluate_reconstruction(model, test_norm)

        train_predictions = [trial.prediction for trial in train_recon.trials]
        test_predictions = [trial.prediction for trial in test_recon.trials]

        train_decoding = evaluate_decoding(train_predictions, train_norm)
        test_decoding = evaluate_decoding(test_predictions, test_norm)
        windowed = evaluate_windowed_decoding(
            test_predictions,
            test_norm,
            fs=subject.fs,
            window_lengths=WINDOW_LENGTHS,
        )

        summary_records.append(
            {
                "subject": subject.subject_id,
                "best_lambda": cv_result.best_lambda,
                "train_corr_mean": train_recon.average_correlation,
                "train_corr_std": float(np.std([t.correlation for t in train_recon.trials])),
                "test_corr_mean": test_recon.average_correlation,
                "test_corr_std": float(np.std([t.correlation for t in test_recon.trials])),
                "train_accuracy": train_decoding.accuracy,
                "test_accuracy": test_decoding.accuracy,
            }
        )

        for length, result in windowed.items():
            window_records.append(
                {
                    "subject": subject.subject_id,
                    "window_length": length,
                    "correct": result.correct,
                    "total": result.total,
                    "accuracy": result.accuracy,
                }
            )

        if not example_ready and subject.subject_id == EXAMPLE_SUBJECT:
            gt = test_norm.wav_attended[EXAMPLE_TRIAL]
            pred = test_predictions[EXAMPLE_TRIAL]
            plot_reconstruction_example(subject.subject_id, subject.fs, gt, pred)
            example_ready = True

    save_csv(
        RESULTS_DIR / "subject_summary.csv",
        summary_records,
        [
            "subject",
            "best_lambda",
            "train_corr_mean",
            "train_corr_std",
            "test_corr_mean",
            "test_corr_std",
            "train_accuracy",
            "test_accuracy",
        ],
    )

    save_csv(
        RESULTS_DIR / "windowed_accuracy.csv",
        window_records,
        ["subject", "window_length", "correct", "total", "accuracy"],
    )

    plot_window_accuracy(window_records)


if __name__ == "__main__":
    main()
