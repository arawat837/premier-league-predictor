"""Unit tests for Current Injuries ingestion, Injury Impact calculation, and Form Upgrades."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import get_current_injuries, get_player_stats
from src.features import (
    FEATURE_COLUMNS,
    calculate_injury_impact,
    calculate_matchup_injury_impact,
    compute_features,
)


def test_get_current_injuries_schema_and_values():
    """Verify get_current_injuries returns valid Premier League team injury records."""
    injuries_df = get_current_injuries()
    assert not injuries_df.empty
    assert "Team" in injuries_df.columns
    assert "Player" in injuries_df.columns
    assert "Status" in injuries_df.columns
    assert "Position" in injuries_df.columns

    # Verify status is strictly normalized to Out or Doubtful
    unique_statuses = set(injuries_df["Status"].unique())
    assert unique_statuses.issubset({"Out", "Doubtful"})

    # Check key teams are present
    teams = set(injuries_df["Team"].unique())
    assert "Arsenal" in teams
    assert "Chelsea" in teams
    assert "Man City" in teams
    assert "Liverpool" in teams


def test_get_player_stats_schema():
    """Verify get_player_stats returns squad player records with minutes and starts."""
    stats_df = get_player_stats()
    assert not stats_df.empty
    assert "Team" in stats_df.columns
    assert "Player" in stats_df.columns
    assert "Minutes" in stats_df.columns
    assert "Starts" in stats_df.columns
    assert "Matches" in stats_df.columns

    # Minutes should be positive integers/floats
    assert (stats_df["Minutes"] >= 0).all()
    assert (stats_df["Starts"] >= 0).all()


def test_calculate_injury_impact_single_team():
    """Verify calculate_injury_impact calculates Missing_Minutes_Share and Missing_Key_Player_Count."""
    injuries = pd.DataFrame([
        {"Team": "TeamA", "Player": "Star Striker", "Status": "Out", "Position": "Forward"},
        {"Team": "TeamA", "Player": "Key Midfielder", "Status": "Doubtful", "Position": "Midfielder"},
        {"Team": "TeamA", "Player": "Bench Player", "Status": "Out", "Position": "Defender"},
    ])

    stats = pd.DataFrame([
        # 3 injured players:
        {"Team": "TeamA", "Player": "Star Striker", "Starts": 8, "Matches": 10, "Minutes": 800, "Start_Pct": 0.8},
        {"Team": "TeamA", "Player": "Key Midfielder", "Starts": 7, "Matches": 10, "Minutes": 700, "Start_Pct": 0.7},
        {"Team": "TeamA", "Player": "Bench Player", "Starts": 2, "Matches": 10, "Minutes": 100, "Start_Pct": 0.2},
        # 8 healthy starters:
        {"Team": "TeamA", "Player": "Healthy 1", "Starts": 10, "Matches": 10, "Minutes": 900, "Start_Pct": 1.0},
        {"Team": "TeamA", "Player": "Healthy 2", "Starts": 10, "Matches": 10, "Minutes": 900, "Start_Pct": 1.0},
        {"Team": "TeamA", "Player": "Healthy 3", "Starts": 10, "Matches": 10, "Minutes": 900, "Start_Pct": 1.0},
        {"Team": "TeamA", "Player": "Healthy 4", "Starts": 10, "Matches": 10, "Minutes": 900, "Start_Pct": 1.0},
        {"Team": "TeamA", "Player": "Healthy 5", "Starts": 10, "Matches": 10, "Minutes": 900, "Start_Pct": 1.0},
        {"Team": "TeamA", "Player": "Healthy 6", "Starts": 10, "Matches": 10, "Minutes": 900, "Start_Pct": 1.0},
        {"Team": "TeamA", "Player": "Healthy 7", "Starts": 10, "Matches": 10, "Minutes": 900, "Start_Pct": 1.0},
        {"Team": "TeamA", "Player": "Healthy 8", "Starts": 10, "Matches": 10, "Minutes": 900, "Start_Pct": 1.0},
        {"Team": "TeamA", "Player": "Healthy 9", "Starts": 10, "Matches": 10, "Minutes": 900, "Start_Pct": 1.0},
    ])
    # Total minutes = 800 + 700 + 100 + (9 * 900) = 1600 + 8100 = 9700
    # Missing minutes = 800 + 700 + 100 = 1600
    # Missing minutes share = 1600 / 9700 = 0.1649

    impact = calculate_injury_impact("TeamA", injuries_df=injuries, player_stats_df=stats)
    assert np.isclose(impact["Missing_Minutes_Share"], 1600.0 / 9700.0, atol=1e-3)
    # Key players (>60% starts): Star Striker (80%) and Key Midfielder (70%). Bench Player (20%) is not.
    assert impact["Missing_Key_Player_Count"] == 2
    # Verify attribute access works on InjuryImpactResult
    assert impact.Missing_Key_Player_Count == 2
    assert np.isclose(impact.Missing_Minutes_Share, 1600.0 / 9700.0, atol=1e-3)


def test_calculate_matchup_injury_impact_delta():
    """Verify Delta_Injury_Impact = Away_Missing_Minutes - Home_Missing_Minutes."""
    injuries = pd.DataFrame([
        {"Team": "HomeFC", "Player": "HomeInjured", "Status": "Out", "Position": "Forward"},
        {"Team": "AwayFC", "Player": "AwayInjured1", "Status": "Out", "Position": "Defender"},
        {"Team": "AwayFC", "Player": "AwayInjured2", "Status": "Out", "Position": "Midfielder"},
    ])

    stats = pd.DataFrame([
        # Home: 1000 total mins, 200 missing -> 0.20
        {"Team": "HomeFC", "Player": "HomeInjured", "Starts": 8, "Matches": 10, "Minutes": 200, "Start_Pct": 0.8},
        {"Team": "HomeFC", "Player": "HomeHealthy", "Starts": 10, "Matches": 10, "Minutes": 800, "Start_Pct": 1.0},
        # Away: 1000 total mins, 500 missing -> 0.50
        {"Team": "AwayFC", "Player": "AwayInjured1", "Starts": 8, "Matches": 10, "Minutes": 300, "Start_Pct": 0.8},
        {"Team": "AwayFC", "Player": "AwayInjured2", "Starts": 7, "Matches": 10, "Minutes": 200, "Start_Pct": 0.7},
        {"Team": "AwayFC", "Player": "AwayHealthy", "Starts": 10, "Matches": 10, "Minutes": 500, "Start_Pct": 1.0},
    ])

    matchup = calculate_matchup_injury_impact("HomeFC", "AwayFC", injuries_df=injuries, player_stats_df=stats)
    # Home missing: 200/1000 = 0.20
    # Away missing: 500/1000 = 0.50
    # Delta = Away - Home = 0.50 - 0.20 = +0.30
    assert np.isclose(matchup["Home_Missing_Minutes"], 0.20)
    assert np.isclose(matchup["Away_Missing_Minutes"], 0.50)
    assert np.isclose(matchup["Delta_Injury_Impact"], 0.30)
    assert matchup["Missing_Key_Player_Count"] == 1  # Home key injuries


def test_form_upgrades_in_features():
    """Verify Opponent_Adjusted_Form_5 and Venue_Specific_Form_5 are computed in compute_features."""
    dates = pd.date_range("2023-08-01", periods=8, freq="7D")
    df = pd.DataFrame({
        "Date": dates,
        "HomeTeam": ["Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal", "Liverpool", "Arsenal", "Chelsea"],
        "AwayTeam": ["Chelsea", "Arsenal", "Liverpool", "Liverpool", "Chelsea", "Arsenal", "Liverpool", "Arsenal"],
        "FTHG": [2, 1, 3, 0, 1, 2, 2, 0],
        "FTAG": [0, 1, 1, 2, 1, 1, 0, 3],
        "FTR":  [2, 1, 2, 0, 1, 1, 2, 0],
        "HS":   [12, 10, 15, 8, 11, 14, 13, 9],
        "AS":   [8, 9, 7, 12, 10, 8, 6, 14],
        "HST":  [5, 4, 6, 2, 4, 5, 5, 3],
        "AST":  [2, 3, 2, 5, 3, 3, 1, 6],
    })

    feat = compute_features(df)
    # Check that required columns exist
    for col in [
        "Delta_Injury_Impact",
        "Home_Injury_Ratio",
        "Away_Injury_Ratio",
        "Venue_Specific_Form_5",
        "Opponent_Adjusted_Form_5",
        "home_venue_form_5",
        "away_venue_form_5",
        "home_opp_adj_form_5",
        "away_opp_adj_form_5",
    ]:
        assert col in feat.columns, f"Missing {col} in computed features"
        assert not feat[col].isna().any(), f"NaN values found in {col}"

    # Verify that required columns are in FEATURE_COLUMNS
    assert "Delta_Injury_Impact" in FEATURE_COLUMNS
    assert "Home_Injury_Ratio" in FEATURE_COLUMNS
    assert "Away_Injury_Ratio" in FEATURE_COLUMNS
    assert "Venue_Specific_Form_5" in FEATURE_COLUMNS
    assert "Opponent_Adjusted_Form_5" in FEATURE_COLUMNS
