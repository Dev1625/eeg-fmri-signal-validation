"""Metrics and temporal null adapted from the original EEG-fMRI analysis."""

import numpy as np


def compute_r(y_true, y_pred):
    """Pearson correlation."""
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return np.corrcoef(y_true, y_pred)[0, 1]


def compute_cv_r2(y_true, y_pred, y_train_mean):
    """Score prediction error against a constant estimated on training data."""
    if len(y_true) == 0:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_train_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def generate_time_shifted_features(features, shift_volumes):
    """Break EEG/BOLD timing while preserving the EEG series up to rotation."""
    return np.roll(features, shift=shift_volumes, axis=0)
