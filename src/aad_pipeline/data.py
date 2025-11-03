from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import scipy.io as sio


EEG_CHANNELS = 64  # use first 64 channels as requested
RESAMPLED_FS = 64  # Hz, provided dataset is down-sampled to 64 Hz


@dataclass
class TrialSet:
    """Container for EEG and speech envelope trials."""

    eeg: List[np.ndarray]
    wav_attended: List[np.ndarray]
    wav_unattended: List[np.ndarray]


@dataclass
class SubjectData:
    subject_id: str
    train: TrialSet
    test: TrialSet
    fs: int = RESAMPLED_FS


def _load_trials(mat_path: Path, key: str) -> TrialSet:
    """Load EEG and speech envelope trials from a MATLAB file."""

    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    struct = mat[key]
    eeg_trials = [np.asarray(trial, dtype=np.float64)[:, :EEG_CHANNELS] for trial in struct.eeg]
    wav_a_trials = [np.asarray(trial, dtype=np.float64).reshape(-1) for trial in struct.wavA]
    wav_b_trials = [np.asarray(trial, dtype=np.float64).reshape(-1) for trial in struct.wavB]
    return TrialSet(eeg=eeg_trials, wav_attended=wav_a_trials, wav_unattended=wav_b_trials)


def load_subject_data(subject_dir: Path) -> SubjectData:
    """Load training and test data for a single subject directory."""

    subject_id = subject_dir.name
    train_path = subject_dir / "train_data.mat"
    test_path = subject_dir / "test_data.mat"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test file for {subject_id}")

    train = _load_trials(train_path, "train_struct")
    test = _load_trials(test_path, "test_struct")
    return SubjectData(subject_id=subject_id, train=train, test=test)


def list_subjects(data_root: Path) -> List[Path]:
    """Return a sorted list of subject directories within the data root."""

    return sorted(
        [p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("S")],
        key=lambda p: int(p.name[1:]),
    )
