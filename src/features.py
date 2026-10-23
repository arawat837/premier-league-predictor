"""Feature engineering module for Premier League match outcome predictor.

Strict zero-leakage implementation:
- All team rolling averages and EWMAs are calculated using .shift(1) grouped by team.
- Pre-match Elo ratings are logged prior to fixture outcome evaluation.
- Match rest days are derived from strictly previous fixture dates.
- Matchup differential features (Delta_Points_Roll_5, Delta_xG_Proxy, Delta_ShotsOnTarget)
  are synthesized from shifted features.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path when script is run directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Default baseline priors for initial games before rolling history is accumulated
DEFAULT_PRIORS = {
    "gf": 1.35,       # average goals scored per team per match
    "ga": 1.35,       # average goals conceded per team per match
    "shots": 12.0,    # average shots per team per match
    "sot": 4.0,       # average shots on target per team per match
    "pts": 1.36,      # average points per team per match (3*0.43 + 1*0.24)
}

STATS_TO_ROLL = ["gf", "ga", "shots", "sot", "pts"]
WINDOWS = [5, 10]


def compute_elo_ratings(
    df: pd.DataFrame,
    initial_elo: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 65.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate pre-match dynamic Elo ratings for Home and Away teams.

    Zero-leakage guarantee:
    For row i, the recorded pre-match Elo rating is taken from the current state
    BEFORE match i's score or result is processed into the rating update.

    Args:
        df: Chronologically sorted DataFrame with HomeTeam, AwayTeam, FTHG, FTAG, FTR.
        initial_elo: Starting Elo rating for unseen teams.
        k_factor: Base Elo update speed.
        home_advantage: Elo points added to home team expectation.

    Returns:
        Tuple of (home_elo, away_elo, elo_diff) series.
    """
    elo_table: Dict[str, float] = {}

    home_elos: List[float] = []
    away_elos: List[float] = []
    elo_diffs: List[float] = []

    for _, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        # Retrieve current pre-match ratings
        r_home = elo_table.get(home_team, initial_elo)
        r_away = elo_table.get(away_team, initial_elo)

        # Record pre-match ratings for this fixture (strictly BEFORE update)
        home_elos.append(r_home)
        away_elos.append(r_away)
        elo_diffs.append(r_home - r_away)

        # Compute match outcome for Elo update
        ftr = row["FTR"]
        # FTR mapping: 2 = Home Win, 1 = Draw, 0 = Away Win
        if ftr == 2:
            s_home, s_away = 1.0, 0.0
        elif ftr == 1:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0

        # Calculate expected scores with home advantage
        dr = (r_home + home_advantage) - r_away
        e_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        e_away = 1.0 - e_home

        # Goal difference margin multiplier
        fthg = row.get("FTHG", 0)
        ftag = row.get("FTAG", 0)
        goal_diff = abs(fthg - ftag)
        if goal_diff <= 1:
            margin_mult = 1.0
        elif goal_diff == 2:
            margin_mult = 1.5
        else:
            margin_mult = 1.75 + (goal_diff - 3) / 8.0

        # Update ratings in elo_table strictly AFTER recording pre-match values
        delta_home = k_factor * margin_mult * (s_home - e_home)
        delta_away = k_factor * margin_mult * (s_away - e_away)

        elo_table[home_team] = r_home + delta_home
        elo_table[away_team] = r_away + delta_away

    return (
        pd.Series(home_elos, index=df.index, name="home_elo"),
        pd.Series(away_elos, index=df.index, name="away_elo"),
        pd.Series(elo_diffs, index=df.index, name="elo_diff"),
    )


def compute_team_rolling_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute 5-game and 10-game rolling averages, EWMAs, and rest days for each team.

    MANDATORY ZERO-LEAKAGE ENFORCEMENT:
    Team match history is constructed chronologically, and every metric is shifted
    via .shift(1) grouped by team before any rolling or EWMA calculation.

    Args:
        df: Standardized match DataFrame.

    Returns:
        Tuple of (home_features_df, away_features_df) keyed by original match index.
    """
    # Build two perspectives per match: home and away
    home_recs = pd.DataFrame({
        "match_idx": df.index,
        "Date": df["Date"],
        "team": df["HomeTeam"],
        "opponent": df["AwayTeam"],
        "is_home": 1,
        "gf": df["FTHG"],
        "ga": df["FTAG"],
        "shots": df["HS"],
        "sot": df["HST"],
        "pts": np.where(df["FTR"] == 2, 3, np.where(df["FTR"] == 1, 1, 0)),
    })

    away_recs = pd.DataFrame({
        "match_idx": df.index,
        "Date": df["Date"],
        "team": df["AwayTeam"],
        "opponent": df["HomeTeam"],
        "is_home": 0,
        "gf": df["FTAG"],
        "ga": df["FTHG"],
        "shots": df["AS"],
        "sot": df["AST"],
        "pts": np.where(df["FTR"] == 0, 3, np.where(df["FTR"] == 1, 1, 0)),
    })

    # Combine into unified team match stream
    team_stream = pd.concat([home_recs, away_recs], ignore_index=True)
    team_stream = team_stream.sort_values(by=["Date", "match_idx"]).reset_index(drop=True)

    # 1. Rest days: strictly difference from previous match Date
    prev_date = team_stream.groupby("team")["Date"].shift(1)
    rest_days = (team_stream["Date"] - prev_date).dt.total_seconds() / (24 * 3600)
    # Default initial match rest to neutral 14 days, clip outliers [1, 45]
    team_stream["rest_days"] = rest_days.fillna(14.0).clip(lower=1.0, upper=45.0)

    # 2. Shifted rolling averages and EWMAs
    for stat in STATS_TO_ROLL:
        # Crucial: shift(1) ensures match i's stat is never visible at match i
        shifted_col = f"{stat}_shifted"
        team_stream[shifted_col] = team_stream.groupby("team")[stat].shift(1)
        prior = DEFAULT_PRIORS.get(stat, 0.0)

        for win in WINDOWS:
            # Rolling average on the shifted series
            roll_col = f"{stat}_roll_{win}"
            roll_series = (
                team_stream.groupby("team")[shifted_col]
                .transform(lambda s: s.rolling(window=win, min_periods=1).mean())
            )
            team_stream[roll_col] = roll_series.fillna(prior)

            # Exponentially Weighted Moving Average (EWMA) on the shifted series
            ewma_col = f"{stat}_ewma_{win}"
            ewma_series = (
                team_stream.groupby("team")[shifted_col]
                .transform(lambda s: s.ewm(span=win, min_periods=1).mean())
            )
            team_stream[ewma_col] = ewma_series.fillna(prior)

    # Separate home and away perspectives back to match_idx
    home_features = team_stream[team_stream["is_home"] == 1].set_index("match_idx")
    away_features = team_stream[team_stream["is_home"] == 0].set_index("match_idx")

    return home_features, away_features


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end feature engineering pipeline for match data.

    Calculates:
    - 5-game & 10-game rolling averages and EWMAs for (GF, GA, Shots, Shots on Target, Points)
    - Dynamic pre-match Elo ratings and Elo differential
    - Pre-match rest days for home and away teams
    - Matchup differential features:
        * Delta_Points_Roll_5
        * Delta_xG_Proxy
        * Delta_ShotsOnTarget

    Args:
        df: Chronologically sorted DataFrame with standardized schema.

    Returns:
        DataFrame augmented with engineered features.
    """
    result = df.copy()

    # 1. Compute dynamic Elo ratings (strictly pre-match)
    home_elo, away_elo, elo_diff = compute_elo_ratings(result)
    result["home_elo"] = home_elo
    result["away_elo"] = away_elo
    result["elo_diff"] = elo_diff

    # 2. Compute team-level rolling and EWMA features
    home_feats, away_feats = compute_team_rolling_features(result)

    # Rest days
    result["home_rest_days"] = home_feats["rest_days"]
    result["away_rest_days"] = away_feats["rest_days"]
    result["delta_rest_days"] = result["home_rest_days"] - result["away_rest_days"]

    # Rolling & EWMA columns
    feature_suffixes: List[str] = []
    for stat in STATS_TO_ROLL:
        for win in WINDOWS:
            feature_suffixes.append(f"{stat}_roll_{win}")
            feature_suffixes.append(f"{stat}_ewma_{win}")

    for suffix in feature_suffixes:
        result[f"home_{suffix}"] = home_feats[suffix]
        result[f"away_{suffix}"] = away_feats[suffix]

    # 3. Differential matchup features
    # Required: Delta_Points_Roll_5
    result["Delta_Points_Roll_5"] = (
        result["home_pts_roll_5"] - result["away_pts_roll_5"]
    )

    # Required: Delta_ShotsOnTarget
    result["Delta_ShotsOnTarget"] = (
        result["home_sot_roll_5"] - result["away_sot_roll_5"]
    )

    # Required: Delta_xG_Proxy
    # Box-score xG proxy: 0.3 * SOT + 0.07 * (Total Shots - SOT)
    home_xg_proxy = (
        0.30 * result["home_sot_roll_5"]
        + 0.07 * (result["home_shots_roll_5"] - result["home_sot_roll_5"]).clip(lower=0)
    )
    away_xg_proxy = (
        0.30 * result["away_sot_roll_5"]
        + 0.07 * (result["away_shots_roll_5"] - result["away_sot_roll_5"]).clip(lower=0)
    )
    result["Delta_xG_Proxy"] = home_xg_proxy - away_xg_proxy

    # Additional high-utility differentials
    result["Delta_Points_Roll_10"] = (
        result["home_pts_roll_10"] - result["away_pts_roll_10"]
    )
    result["Delta_Points_EWMA_5"] = (
        result["home_pts_ewma_5"] - result["away_pts_ewma_5"]
    )
    result["Delta_GoalsFor_Roll_5"] = (
        result["home_gf_roll_5"] - result["away_gf_roll_5"]
    )
    result["Delta_GoalsAgainst_Roll_5"] = (
        result["home_ga_roll_5"] - result["away_ga_roll_5"]
    )
    result["Delta_Shots_Roll_5"] = (
        result["home_shots_roll_5"] - result["away_shots_roll_5"]
    )

    return result


# Core list of pre-match engineered feature column names
FEATURE_COLUMNS = [
    # Elo
    "home_elo",
    "away_elo",
    "elo_diff",
    # Rest days
    "home_rest_days",
    "away_rest_days",
    "delta_rest_days",
    # Differentials (explicitly requested)
    "Delta_Points_Roll_5",
    "Delta_xG_Proxy",
    "Delta_ShotsOnTarget",
    # Additional differentials
    "Delta_Points_Roll_10",
    "Delta_Points_EWMA_5",
    "Delta_GoalsFor_Roll_5",
    "Delta_GoalsAgainst_Roll_5",
    "Delta_Shots_Roll_5",
] + [
    f"home_{stat}_{kind}_{win}"
    for stat in STATS_TO_ROLL
    for kind in ["roll", "ewma"]
    for win in WINDOWS
] + [
    f"away_{stat}_{kind}_{win}"
    for stat in STATS_TO_ROLL
    for kind in ["roll", "ewma"]
    for win in WINDOWS
]


def build_and_save_features(
    raw_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build features from all standardized raw data and save to parquet.

    Args:
        raw_dir: Directory containing raw match CSVs.
        output_path: Target parquet destination.

    Returns:
        Processed DataFrame with engineered features.
    """
    from src.data_loader import get_data

    if output_path is None:
        output_path = Path("data") / "processed" / "features.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading and standardizing raw match data...")
    raw_df = get_data(raw_dir=raw_dir)
    print(f"Loaded {len(raw_df)} fixtures.")

    print("Computing leak-free features...")
    featured_df = compute_features(raw_df)

    print(f"Saving {featured_df.shape} dataset to {output_path}...")
    featured_df.to_parquet(output_path, index=False)
    print(f"Successfully saved {output_path} ({output_path.stat().st_size:,} bytes).")

    return featured_df


if __name__ == "__main__":
    print("=" * 60)
    print("Premier League Feature Engineering & Parquet Export (Phase 2)")
    print("=" * 60)
    df_feat = build_and_save_features()
    print("\nFeature Summary:")
    print(f"Total Rows:    {len(df_feat):,}")
    print(f"Total Columns: {df_feat.shape[1]}")
    print(f"Total Pre-Match Features: {len(FEATURE_COLUMNS)}")
    print(f"Null Values in Features: {df_feat[FEATURE_COLUMNS].isna().sum().sum()}")
    print("\nSample Engineered Columns (first 3 rows):")
    sample_cols = [
        "Date", "HomeTeam", "AwayTeam", "FTR",
        "home_elo", "away_elo", "elo_diff",
        "Delta_Points_Roll_5", "Delta_xG_Proxy", "Delta_ShotsOnTarget"
    ]
    print(df_feat[sample_cols].head(3))
    print("=" * 60)
