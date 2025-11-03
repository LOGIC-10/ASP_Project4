"""Utilities for Project 4 auditory attention decoding pipeline."""

from .data import load_subject_data, SubjectData
from .preprocessing import compute_normalization, apply_normalization, NormalizationStats
from .model import cross_validate_lambda, train_final_model, CrossValidationResult
from .evaluation import (
    evaluate_reconstruction,
    evaluate_decoding,
    evaluate_windowed_decoding,
)
from .metrics import pearsonr_np

__all__ = [
    "load_subject_data",
    "SubjectData",
    "compute_normalization",
    "apply_normalization",
    "NormalizationStats",
    "cross_validate_lambda",
    "CrossValidationResult",
    "train_final_model",
    "evaluate_reconstruction",
    "evaluate_decoding",
    "evaluate_windowed_decoding",
    "pearsonr_np",
]
