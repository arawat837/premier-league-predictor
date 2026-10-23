"""Leakage tests to ensure pre-match features never incorporate current or future match statistics."""

import pandas as pd
import numpy as np
import pytest

from src.features import compute_features, FEATURE_COLUMNS


@pytest.fixture
def sample_match_history():
    """Create a synthetic chronological match history with repeated team matchups."""
    data = {
        "Date": [
            "2023-08-12", "2023-08-13",
            "2023-08-19", "2023-08-20",
            "2023-08-26", "2023-08-27",
            "2023-09-02", "2023-09-03",
            "2023-09-16", "2023-09-17",
            "2023-09-23", "2023-09-24",
        ],
        "HomeTeam": [
            "Arsenal", "Chelsea",
            "Liverpool", "Man City",
            "Arsenal", "Chelsea",
            "Liverpool", "Man City",
            "Arsenal", "Chelsea",
            "Liverpool", "Man City",
        ],
        "AwayTeam": [
            "Liverpool", "Man City",
            "Arsenal", "Chelsea",
            "Chelsea", "Liverpool",
            "Arsenal", "Chelsea",
            "Man City", "Arsenal",
            "Man City", "Liverpool",
        ],
        "FTHG": [2, 1, 3, 2, 1, 0, 2, 4, 1, 2, 2, 3],
        "FTAG": [1, 1, 1, 0, 1, 2, 2, 1, 3, 1, 1, 0],
        "FTR":  [2, 1, 2, 2, 1, 0, 1, 2, 0, 2, 2, 2],
        "HS":   [14, 10, 18, 15, 12, 8, 16, 20, 11, 13, 15, 19],
        "AS":   [9, 12, 8, 6, 11, 14, 13, 7, 15, 10, 9, 5],
        "HST":  [6, 4, 7, 5, 4, 2, 5, 9, 3, 5, 6, 8],
        "AST":  [3, 4, 2, 1, 4, 6, 4, 2, 7, 3, 3, 1],
    }
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_zero_leakage_on_current_match_modification(sample_match_history):
    """Assert that modifying row i's actual match score or match stats produces

    zero change in row i's pre-match rolling and differential features.
    """
    df_orig = sample_match_history.copy()
    features_orig = compute_features(df_orig)

    # Pick row i where teams already have historical matches (e.g., row 6: Liverpool vs Arsenal)
    target_row_idx = 6

    # Create a corrupted copy where row i's match outcome and stats are wildly changed
    df_corrupted = sample_match_history.copy()
    df_corrupted.loc[target_row_idx, "FTHG"] = 99
    df_corrupted.loc[target_row_idx, "FTAG"] = 88
    df_corrupted.loc[target_row_idx, "FTR"] = 0  # was 1 (Draw), now 0 (Away Win)
    df_corrupted.loc[target_row_idx, "HS"] = 250
    df_corrupted.loc[target_row_idx, "AS"] = 300
    df_corrupted.loc[target_row_idx, "HST"] = 100
    df_corrupted.loc[target_row_idx, "AST"] = 120

    features_corrupted = compute_features(df_corrupted)

    # For the target row i, extract all pre-match feature columns
    orig_row_features = features_orig.loc[target_row_idx, FEATURE_COLUMNS]
    corrupted_row_features = features_corrupted.loc[target_row_idx, FEATURE_COLUMNS]

    # Verify that modifying row i produced ZERO change in row i's pre-match features
    pd.testing.assert_series_equal(
        orig_row_features,
        corrupted_row_features,
        check_dtype=True,
        check_names=True,
        obj="Row i pre-match features",
    )


def test_leakage_check_across_multiple_rows(sample_match_history):
    """Test every row after initial matches to ensure zero leakage universally."""
    df_orig = sample_match_history.copy()
    features_orig = compute_features(df_orig)

    # Test rows 4 through 11
    for test_idx in range(4, len(sample_match_history)):
        df_mod = sample_match_history.copy()
        # Flip result and change goals
        df_mod.loc[test_idx, "FTHG"] += 10
        df_mod.loc[test_idx, "FTAG"] += 5
        df_mod.loc[test_idx, "FTR"] = (df_mod.loc[test_idx, "FTR"] + 1) % 3
        df_mod.loc[test_idx, "HST"] += 20
        df_mod.loc[test_idx, "AST"] += 15

        features_mod = compute_features(df_mod)

        orig_features = features_orig.loc[test_idx, FEATURE_COLUMNS]
        mod_features = features_mod.loc[test_idx, FEATURE_COLUMNS]

        pd.testing.assert_series_equal(
            orig_features,
            mod_features,
            check_dtype=False,
            obj=f"Row {test_idx} features under current match mutation",
        )


def test_proper_forward_propagation(sample_match_history):
    """Ensure that modifying row i's outcome DOES alter subsequent matches for those teams,

    confirming the features are active and not static stubs.
    """
    df_orig = sample_match_history.copy()
    features_orig = compute_features(df_orig)

    # Modify row 0 (Arsenal vs Liverpool)
    df_mod = sample_match_history.copy()
    df_mod.loc[0, "FTHG"] = 10
    df_mod.loc[0, "HST"] = 20

    features_mod = compute_features(df_mod)

    # In row 2 (Liverpool vs Arsenal, the next time Arsenal/Liverpool play):
    # Arsenal's pre-match rolling goals against or Liverpool's pre-match goals against should differ
    orig_row2 = features_orig.loc[2, FEATURE_COLUMNS]
    mod_row2 = features_mod.loc[2, FEATURE_COLUMNS]

    assert not orig_row2.equals(mod_row2), "Modifying row 0 should affect row 2's pre-match features!"
