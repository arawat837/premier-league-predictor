"""End-to-end smoke tests for the model pipeline and dashboard inference."""

import numpy as np
import pandas as pd

from app.dashboard import (
    compute_matchup_features,
    compute_shap_explanations,
    load_calibrated_model,
    load_historical_data,
)
from src.data_loader import get_upcoming_fixtures
from src.features import FEATURE_COLUMNS
from src.models import _normalize_probabilities
from streamlit.testing.v1 import AppTest


def test_get_upcoming_fixtures():
    """Verify that get_upcoming_fixtures returns valid scheduled fixtures."""
    df_upcoming = get_upcoming_fixtures()
    assert not df_upcoming.empty
    assert "HomeTeam" in df_upcoming.columns
    assert "AwayTeam" in df_upcoming.columns
    assert "Date" in df_upcoming.columns
    assert len(df_upcoming) >= 10


def test_upcoming_gameweek_apptest():
    """Verify Streamlit AppTest renders both Custom Matchup and Upcoming Gameweek tabs without exceptions."""
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent / "app" / "dashboard.py"
    at = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not at.exception
    assert len(at.tabs) == 2


def test_end_to_end_arsenal_vs_chelsea_inference():
    """Verify end-to-end prediction and SHAP explanation for Arsenal vs. Chelsea."""
    df = load_historical_data()
    model = load_calibrated_model()

    feat_row = compute_matchup_features(df, "Arsenal", "Chelsea")
    X_in = pd.DataFrame([feat_row[FEATURE_COLUMNS].to_dict()])

    probs = _normalize_probabilities(model.predict_proba(X_in))[0]
    assert probs.shape == (3,)
    assert np.isclose(np.sum(probs), 1.0)
    assert np.all((probs > 0.0) & (probs < 1.0))

    favored_class = int(np.argmax(probs))
    shap_df = compute_shap_explanations(model, feat_row, target_class=favored_class)
    assert len(shap_df) == len(FEATURE_COLUMNS)
    assert "SHAP_Impact" in shap_df.columns
