"""Unit tests for walk-forward splitting, calibration, and evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate_high_conviction, multiclass_brier_score
from src.models import walk_forward_expanding_splits


def test_walk_forward_splits_strictly_chronological():
    """Verify that walk-forward splits never shuffle and train strictly precedes test."""
    dummy_df = pd.DataFrame({"x": np.arange(1200)})
    splits = list(walk_forward_expanding_splits(dummy_df, initial_train_size=400, step_size=300))

    assert len(splits) == 3

    prev_train_size = 0
    for fold, train_idx, test_idx in splits:
        # Strict chronological ordering
        assert np.max(train_idx) < np.min(test_idx)
        # Contiguous indices with zero shuffling
        assert np.array_equal(train_idx, np.arange(0, len(train_idx)))
        assert np.array_equal(test_idx, np.arange(len(train_idx), len(train_idx) + len(test_idx)))
        # Expanding window check
        assert len(train_idx) > prev_train_size
        prev_train_size = len(train_idx)


def test_multiclass_brier_score_and_high_conviction():
    """Verify Brier score calculation and high-conviction filtering at max(P) >= 0.65."""
    y_true = np.array([2, 0, 1, 2])
    y_prob = np.array([
        [0.10, 0.15, 0.75],  # Correct, max(P)=0.75 >= 0.65
        [0.70, 0.20, 0.10],  # Correct, max(P)=0.70 >= 0.65
        [0.30, 0.40, 0.30],  # Correct, max(P)=0.40 < 0.65
        [0.45, 0.35, 0.20],  # Wrong,   max(P)=0.45 < 0.65
    ])

    bs = multiclass_brier_score(y_true, y_prob)
    assert 0.0 <= bs <= 2.0

    hc = evaluate_high_conviction(y_true, y_prob, threshold=0.65)
    assert hc["total_matches"] == 4
    assert hc["qualified_matches"] == 2
    assert np.isclose(hc["coverage"], 0.5)
    assert np.isclose(hc["accuracy"], 1.0)
