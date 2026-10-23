"""Streamlit interactive dashboard for Premier League Match Outcome & Probability Predictor.

Features:
- Club dropdown selectors for Home Team and Away Team (defaults to Arsenal vs. Chelsea)
- Real-time retrieval of latest team rolling stats, EWMAs, Elo ratings, and matchup differentials
- Calibrated probability predictions (Home Win, Draw, Away Win) with horizontal probability bars
- Prominent "High-Conviction Lock" badge when max(P) >= 65% ( plus league-wide High-Conviction Lock scanner)
- SHAP TreeExplainer attribution displaying the Top 3 driving features and interactive SHAP summary chart
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_upcoming_fixtures
from src.features import FEATURE_COLUMNS, compute_features
from src.models import _normalize_probabilities

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_xgb.joblib"
OOF_PATH = PROJECT_ROOT / "data" / "processed" / "oof_predictions.parquet"

OUTCOME_NAMES = {
    0: "Away Win",
    1: "Draw",
    2: "Home Win",
}

FEATURE_LABELS = {
    "elo_diff": "Elo Rating Differential (Home - Away)",
    "home_elo": "Home Team Elo Rating",
    "away_elo": "Away Team Elo Rating",
    "Delta_Points_Roll_5": "5-Match Form Points Differential",
    "Delta_Points_Roll_10": "10-Match Form Points Differential",
    "Delta_Points_EWMA_5": "5-Match EWMA Points Differential",
    "Delta_xG_Proxy": "Expected Goals (xG Proxy) Differential",
    "Delta_ShotsOnTarget": "Shots on Target Differential (Last 5)",
    "Delta_GoalsFor_Roll_5": "Goals Scored Differential (Last 5)",
    "Delta_GoalsAgainst_Roll_5": "Goals Conceded Differential (Last 5)",
    "Delta_Shots_Roll_5": "Total Shots Differential (Last 5)",
    "home_pts_roll_5": "Home Team Points / Game (Last 5)",
    "away_pts_roll_5": "Away Team Points / Game (Last 5)",
    "home_pts_ewma_5": "Home Team EWMA Points (Span 5)",
    "away_pts_ewma_5": "Away Team EWMA Points (Span 5)",
    "home_gf_roll_5": "Home Team Goals Scored / Game (Last 5)",
    "away_gf_roll_5": "Away Team Goals Scored / Game (Last 5)",
    "home_ga_roll_5": "Home Team Goals Conceded / Game (Last 5)",
    "away_ga_roll_5": "Away Team Goals Conceded / Game (Last 5)",
    "home_sot_roll_5": "Home Team Shots on Target (Last 5)",
    "away_sot_roll_5": "Away Team Shots on Target (Last 5)",
    "home_shots_roll_5": "Home Team Total Shots (Last 5)",
    "away_shots_roll_5": "Away Team Total Shots (Last 5)",
    "delta_rest_days": "Rest Days Differential",
}


def format_feature_name(col: str) -> str:
    """Convert raw feature column identifier into a clean human-readable label."""
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]
    return (
        col.replace("home_", "Home ")
        .replace("away_", "Away ")
        .replace("_roll_", " Rolling ")
        .replace("_ewma_", " EWMA ")
        .replace("gf", "Goals For")
        .replace("ga", "Goals Against")
        .replace("sot", "Shots on Target")
        .replace("pts", "Points")
        .replace("_", " ")
        .title()
    )


@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """Load processed historical match dataset, generating it if not yet present."""
    if not FEATURES_PATH.exists():
        from src.features import build_and_save_features
        build_and_save_features(output_path=FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_values(by=["Date", "HomeTeam"]).reset_index(drop=True)
    return df


@st.cache_resource
def load_calibrated_model() -> Any:
    """Load trained CalibratedClassifierCV XGBoost model pipeline, training if not yet present."""
    if not MODEL_PATH.exists():
        from src.models import train_and_evaluate_all
        train_and_evaluate_all(model_output_path=MODEL_PATH)
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_upcoming_fixtures() -> pd.DataFrame:
    """Fetch or load upcoming Premier League gameweek fixtures."""
    return get_upcoming_fixtures()


@st.cache_data
def get_current_pl_clubs(df: pd.DataFrame) -> List[str]:
    """Return sorted list of current Premier League clubs from the latest season."""
    latest_cutoff = df["Date"].max() - pd.Timedelta(days=365)
    recent_df = df[df["Date"] >= latest_cutoff]
    clubs = sorted(set(recent_df["HomeTeam"]).union(set(recent_df["AwayTeam"])))
    return clubs


def compute_matchup_features(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> pd.Series:
    """Retrieve latest rolling stats for both teams and compute the pre-match feature vector.

    Appends a virtual upcoming fixture strictly after the latest historical date and runs
    the leak-free feature engineering pipeline so that all .shift(1) windows and post-match
    Elo updates reflect each team's most recent completed fixtures.
    """
    base_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST"]
    hist_raw = df[base_cols].copy()

    upcoming_date = hist_raw["Date"].max() + pd.Timedelta(days=7)
    upcoming_row = pd.DataFrame([{
        "Date": upcoming_date,
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "FTHG": 0,
        "FTAG": 0,
        "FTR": 1,
        "HS": 0,
        "AS": 0,
        "HST": 0,
        "AST": 0,
    }])

    combined = pd.concat([hist_raw, upcoming_row], ignore_index=True)
    featured = compute_features(combined)
    return featured.iloc[-1]


def compute_shap_explanations(
    calibrated_model: Any,
    feature_row: pd.Series,
    target_class: int,
) -> pd.DataFrame:
    """Compute SHAP values for the predicted outcome class across calibrated XGBoost estimators.

    Args:
        calibrated_model: Trained CalibratedClassifierCV wrapping XGBoost pipelines.
        feature_row: Series containing FEATURE_COLUMNS for the selected matchup.
        target_class: Integer class index (0=Away, 1=Draw, 2=Home) to explain.

    Returns:
        DataFrame with Feature, Label, Value, and SHAP impact sorted by magnitude.
    """
    X_df = pd.DataFrame([feature_row[FEATURE_COLUMNS].to_dict()])

    fold_shap_vectors: List[np.ndarray] = []
    for cal_clf in calibrated_model.calibrated_classifiers_:
        pipe = cal_clf.estimator
        imputer = pipe.named_steps["imputer"]
        scaler = pipe.named_steps["scaler"]
        xgb_clf = pipe.named_steps["clf"]

        X_imp = imputer.transform(X_df)
        X_scaled = scaler.transform(X_imp)

        explainer = shap.TreeExplainer(xgb_clf)
        sv = explainer.shap_values(X_scaled)

        # Handle shape variations across SHAP versions: (1, n_features, 3) or list of 3 arrays
        if isinstance(sv, list):
            class_sv = np.asarray(sv[target_class])[0]
        else:
            sv_arr = np.asarray(sv)
            if sv_arr.ndim == 3:
                class_sv = sv_arr[0, :, target_class]
            else:
                class_sv = sv_arr[0]

        fold_shap_vectors.append(class_sv)

    mean_shap = np.mean(np.vstack(fold_shap_vectors), axis=0)

    shap_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Label": [format_feature_name(c) for c in FEATURE_COLUMNS],
        "Value": [float(feature_row[c]) for c in FEATURE_COLUMNS],
        "SHAP_Impact": mean_shap,
        "Abs_Impact": np.abs(mean_shap),
    })

    # Sort by positive contribution toward the favored outcome first, then magnitude
    shap_df = shap_df.sort_values(by="SHAP_Impact", ascending=False).reset_index(drop=True)
    return shap_df


def render_probability_bar(label: str, prob: float, color: str, is_favored: bool) -> None:
    """Render a styled horizontal probability bar with clean percentage readout."""
    pct = prob * 100.0
    border_style = f"2px solid {color}" if is_favored else "1px solid #334155"
    bg_card = "rgba(30, 41, 59, 0.85)" if is_favored else "rgba(15, 23, 42, 0.65)"
    badge = (
        f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:99px;'
        f'font-size:0.75rem;margin-left:8px;font-weight:700;">FAVORED</span>'
        if is_favored
        else ""
    )

    st.markdown(
        f"""
        <div style="background:{bg_card};border:{border_style};border-radius:10px;padding:14px 18px;margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="font-size:1.05rem;font-weight:600;color:#f8fafc;">{label} {badge}</span>
                <span style="font-size:1.25rem;font-weight:800;color:{color};">{pct:.1f}%</span>
            </div>
            <div style="width:100%;background:#1e293b;height:16px;border-radius:999px;overflow:hidden;">
                <div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Premier League Match Outcome & Probability Predictor",
        page_icon="⚽",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .lock-badge {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            color: white;
            padding: 16px 24px;
            border-radius: 12px;
            border: 2px solid #34d399;
            box-shadow: 0 8px 20px rgba(16, 185, 129, 0.35);
            text-align: center;
            margin: 12px 0 20px 0;
        }
        .standard-badge {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: #e2e8f0;
            padding: 14px 20px;
            border-radius: 12px;
            border: 1px solid #475569;
            text-align: center;
            margin: 12px 0 20px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚽ Premier League Match Outcome & Probability Predictor")
    st.caption(
        "Zero-Leakage Walk-Forward XGBoost (`multi:softprob`) + Isotonic Probability Calibration (`CalibratedClassifierCV`)"
    )

    df = load_historical_data()
    model = load_calibrated_model()
    clubs = get_current_pl_clubs(df)

    # Support URL query params or default to Arsenal vs. Chelsea
    query_params = st.query_params
    default_home = query_params.get("home", "Arsenal")
    default_away = query_params.get("away", "Chelsea")

    home_idx = clubs.index(default_home) if default_home in clubs else 0
    away_idx = clubs.index(default_away) if default_away in clubs else min(1, len(clubs) - 1)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Model & Filter Settings")
        lock_threshold = st.slider(
            "High-Conviction Lock Threshold",
            min_value=0.50,
            max_value=0.75,
            value=0.65,
            step=0.01,
            help="Fixtures where max(P) meets or exceeds this threshold trigger the High-Conviction Lock badge.",
        )
        st.markdown("---")
        st.subheader("📊 Walk-Forward Validation")
        st.metric("Out-of-Fold Test Matches", "1,520 (4 Seasons)")
        st.metric("XGBoost Multi Log Loss", "1.0027 (Cal: 1.0141)")
        st.metric("Multiclass Brier Score", "0.5975 (Cal: 0.6022)")
        st.metric("High-Conviction Precision", "70.9% (Raw) / 67.0% (Cal)")

    # Create Tabs: Match Simulator & Upcoming Gameweek
    tab_simulator, tab_upcoming = st.tabs(["🎯 Custom Matchup Simulator", "📅 Upcoming Gameweek"])

    with tab_simulator:
        # Club Selection Dropdowns
        col_select1, col_select2 = st.columns(2)
        with col_select1:
            home_team = st.selectbox("🏟️ Select Home Club", options=clubs, index=home_idx, key="home_club")
        with col_select2:
            away_team = st.selectbox("✈️ Select Away Club", options=clubs, index=away_idx, key="away_club")

        if home_team == away_team:
            st.warning("Please select two different Premier League clubs to simulate a fixture.")
        else:
            # Compute pre-match feature vector and calibrated probabilities
            feature_row = compute_matchup_features(df, home_team, away_team)
            X_input = pd.DataFrame([feature_row[FEATURE_COLUMNS].to_dict()])
            raw_probs = model.predict_proba(X_input)
            probs = _normalize_probabilities(raw_probs)[0]

            prob_away, prob_draw, prob_home = float(probs[0]), float(probs[1]), float(probs[2])
            max_prob = float(np.max(probs))
            favored_class = int(np.argmax(probs))
            favored_name = (
                f"{home_team} (Home Win)"
                if favored_class == 2
                else (f"{away_team} (Away Win)" if favored_class == 0 else "Draw")
            )

            st.markdown("---")

            # High-Conviction Lock Badge Check (Threshold default = 65%)
            if max_prob >= lock_threshold:
                st.markdown(
                    f"""
                    <div class="lock-badge">
                        <div style="font-size:0.9rem;letter-spacing:1.5px;text-transform:uppercase;font-weight:700;">
                            🔒 HIGH-CONVICTION LOCK QUALIFIED (P &ge; {lock_threshold*100:.0f}%)
                        </div>
                        <div style="font-size:1.65rem;font-weight:800;margin-top:4px;">
                            {favored_name} — {max_prob*100:.1f}% Calibrated Win Probability
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="standard-badge">
                        <div style="font-size:0.85rem;letter-spacing:1px;text-transform:uppercase;color:#94a3b8;font-weight:600;">
                            ⚖️ STANDARD CONVICTION FIXTURE (Max Probability: {max_prob*100:.1f}% &lt; {lock_threshold*100:.0f}% Lock Threshold)
                        </div>
                        <div style="font-size:1.3rem;font-weight:700;margin-top:4px;color:#f8fafc;">
                            Model Lean: {favored_name} ({max_prob*100:.1f}%)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Main Split: Left = Probability Bars & Team Stats | Right = Top 3 SHAP Drivers & Chart
            left_col, right_col = st.columns([1.05, 1.15], gap="large")

            with left_col:
                st.subheader(f"📈 Calibrated Match Probabilities: {home_team} vs. {away_team}")

                render_probability_bar(
                    label=f"🏠 Home Win ({home_team})",
                    prob=prob_home,
                    color="#3b82f6",
                    is_favored=(favored_class == 2),
                )
                render_probability_bar(
                    label="🤝 Draw",
                    prob=prob_draw,
                    color="#f59e0b",
                    is_favored=(favored_class == 1),
                )
                render_probability_bar(
                    label=f"✈️ Away Win ({away_team})",
                    prob=prob_away,
                    color="#ef4444",
                    is_favored=(favored_class == 0),
                )

                st.subheader("📋 Latest Pre-Match Rolling & Elo Snapshot")
                snapshot_df = pd.DataFrame({
                    "Metric": [
                        "Pre-Match Elo Rating",
                        "Points / Game (Last 5)",
                        "Goals Scored / Game (Last 5)",
                        "Goals Conceded / Game (Last 5)",
                        "Shots on Target / Game (Last 5)",
                        "Total Shots / Game (Last 5)",
                    ],
                    f"{home_team} (Home)": [
                        f"{feature_row['home_elo']:.1f}",
                        f"{feature_row['home_pts_roll_5']:.2f}",
                        f"{feature_row['home_gf_roll_5']:.2f}",
                        f"{feature_row['home_ga_roll_5']:.2f}",
                        f"{feature_row['home_sot_roll_5']:.2f}",
                        f"{feature_row['home_shots_roll_5']:.2f}",
                    ],
                    f"{away_team} (Away)": [
                        f"{feature_row['away_elo']:.1f}",
                        f"{feature_row['away_pts_roll_5']:.2f}",
                        f"{feature_row['away_gf_roll_5']:.2f}",
                        f"{feature_row['away_ga_roll_5']:.2f}",
                        f"{feature_row['away_sot_roll_5']:.2f}",
                        f"{feature_row['away_shots_roll_5']:.2f}",
                    ],
                    "Differential (H - A)": [
                        f"{feature_row['elo_diff']:+.1f}",
                        f"{feature_row['Delta_Points_Roll_5']:+.2f}",
                        f"{feature_row['Delta_GoalsFor_Roll_5']:+.2f}",
                        f"{feature_row['Delta_GoalsAgainst_Roll_5']:+.2f}",
                        f"{feature_row['Delta_ShotsOnTarget']:+.2f}",
                        f"{feature_row['Delta_Shots_Roll_5']:+.2f}",
                    ],
                })
                st.dataframe(snapshot_df, width="stretch", hide_index=True)

            with right_col:
                st.subheader(f"🧠 Top 3 SHAP Driving Features Favoring {favored_name}")
                shap_df = compute_shap_explanations(model, feature_row, target_class=favored_class)
                top_3 = shap_df.head(3)

                for idx, row in top_3.iterrows():
                    rank = idx + 1
                    impact = row["SHAP_Impact"]
                    val = row["Value"]
                    st.markdown(
                        f"""
                        <div style="background:rgba(30,41,59,0.7);border-left:4px solid #10b981;padding:12px 16px;border-radius:8px;margin-bottom:10px;">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <span style="font-weight:700;font-size:1.0rem;color:#f8fafc;">
                                    #{rank} {row['Label']}
                                </span>
                                <span style="background:rgba(16,185,129,0.2);color:#34d399;padding:3px 10px;border-radius:6px;font-weight:700;font-size:0.85rem;">
                                    +{impact:.3f} SHAP log-odds
                                </span>
                            </div>
                            <div style="color:#94a3b8;font-size:0.88rem;margin-top:4px;">
                                Feature code: <code>{row['Feature']}</code> &bull; Current pre-match value: <strong>{val:.2f}</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.subheader("📊 SHAP Feature Attribution Summary (Top 10 by Magnitude)")
                top_10_abs = shap_df.sort_values(by="Abs_Impact", ascending=True).tail(10)
                chart_data = pd.DataFrame({
                    "Feature": top_10_abs["Label"],
                    "SHAP Impact": top_10_abs["SHAP_Impact"],
                }).set_index("Feature")
                st.bar_chart(chart_data, horizontal=True, color="#3b82f6")

    with tab_upcoming:
        st.subheader("📅 Next Premier League Gameweek Schedule & Predictions")
        st.caption(
            "Automatically calculates pre-match feature vectors using each club's latest rolling stats and identifies High-Conviction Locks."
        )

        upcoming_df = load_upcoming_fixtures()
        if upcoming_df is None or upcoming_df.empty:
            st.info("No upcoming fixtures scheduled currently.")
        else:
            fixture_results: List[Dict[str, Any]] = []
            for _, row in upcoming_df.iterrows():
                h_team = row["HomeTeam"]
                a_team = row["AwayTeam"]
                m_date = row.get("Date", "")
                m_day = row.get("Matchday", "")

                try:
                    f_row = compute_matchup_features(df, h_team, a_team)
                    X_fix = pd.DataFrame([f_row[FEATURE_COLUMNS].to_dict()])
                    p_fix = _normalize_probabilities(model.predict_proba(X_fix))[0]
                    p_away, p_draw, p_home = float(p_fix[0]), float(p_fix[1]), float(p_fix[2])
                    max_p = float(np.max(p_fix))
                    fav_cls = int(np.argmax(p_fix))
                    is_lock = max_p >= lock_threshold
                    fav_name = (
                        f"{h_team} (Home Win)"
                        if fav_cls == 2
                        else (f"{a_team} (Away Win)" if fav_cls == 0 else "Draw")
                    )

                    fixture_results.append({
                        "Matchday": m_day,
                        "Date": m_date,
                        "HomeTeam": h_team,
                        "AwayTeam": a_team,
                        "prob_home": p_home,
                        "prob_draw": p_draw,
                        "prob_away": p_away,
                        "max_prob": max_p,
                        "favored_class": fav_cls,
                        "favored_name": fav_name,
                        "is_lock": is_lock,
                        "home_elo": f_row["home_elo"],
                        "away_elo": f_row["away_elo"],
                        "elo_diff": f_row["elo_diff"],
                        "Delta_Points_Roll_5": f_row["Delta_Points_Roll_5"],
                        "Delta_xG_Proxy": f_row["Delta_xG_Proxy"],
                    })
                except Exception as exc:
                    st.warning(f"Could not compute features for {h_team} vs {a_team}: {exc}")

            total_fixtures = len(fixture_results)
            lock_count = sum(1 for f in fixture_results if f["is_lock"])
            avg_conviction = (
                float(np.mean([f["max_prob"] for f in fixture_results])) * 100.0 if fixture_results else 0.0
            )

            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Total Scheduled Matches", f"{total_fixtures}")
            with mcol2:
                st.metric(
                    "🔒 High-Conviction Locks",
                    f"{lock_count} / {total_fixtures}",
                    help=f"Matches where max win probability >= {lock_threshold*100:.0f}%",
                )
            with mcol3:
                st.metric("Average Model Top Pick Probability", f"{avg_conviction:.1f}%")

            st.markdown("---")

            # Display View toggle
            view_mode = st.radio(
                "Display View:",
                ["Fixture Cards (Interactive)", "Gameweek Summary Table"],
                horizontal=True,
                key="upcoming_view",
            )

            if view_mode == "Gameweek Summary Table":
                table_rows = []
                for fix in fixture_results:
                    lock_str = (
                        f"🔒 LOCK ({fix['favored_name']} {fix['max_prob']*100:.1f}%)"
                        if fix["is_lock"]
                        else "Standard"
                    )
                    table_rows.append({
                        "Matchday": fix["Matchday"],
                        "Kickoff": fix["Date"],
                        "Fixture": f"{fix['HomeTeam']} vs. {fix['AwayTeam']}",
                        "Home Win %": f"{fix['prob_home']*100:.1f}%",
                        "Draw %": f"{fix['prob_draw']*100:.1f}%",
                        "Away Win %": f"{fix['prob_away']*100:.1f}%",
                        "Model Lean": f"{fix['favored_name']} ({fix['max_prob']*100:.1f}%)",
                        "Elo Diff": f"{fix['elo_diff']:+.0f}",
                        "Lock Status": lock_str,
                    })
                st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
            else:
                for i in range(0, len(fixture_results), 2):
                    c1, c2 = st.columns(2)
                    pair = [fixture_results[i]]
                    if i + 1 < len(fixture_results):
                        pair.append(fixture_results[i + 1])

                    for col_obj, fix in zip([c1, c2], pair):
                        with col_obj:
                            border_color = "#10b981" if fix["is_lock"] else "#334155"
                            bg_color = (
                                "rgba(16, 185, 129, 0.08)"
                                if fix["is_lock"]
                                else "rgba(30, 41, 59, 0.7)"
                            )
                            lock_badge_html = (
                                f'<span style="background:#059669;color:#ffffff;padding:3px 10px;border-radius:6px;font-size:0.8rem;font-weight:700;">'
                                f'🔒 HIGH-CONVICTION LOCK ({fix["max_prob"]*100:.1f}%)</span>'
                                if fix["is_lock"]
                                else f'<span style="background:#334155;color:#94a3b8;padding:3px 8px;border-radius:6px;font-size:0.75rem;">Standard</span>'
                            )

                            st.markdown(
                                f"""
                                <div style="background:{bg_color};border:1.5px solid {border_color};border-radius:12px;padding:16px;margin-bottom:14px;">
                                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                                        <span style="color:#94a3b8;font-size:0.82rem;font-weight:600;">Matchday {fix['Matchday']} &bull; {fix['Date']}</span>
                                        {lock_badge_html}
                                    </div>
                                    <div style="font-size:1.15rem;font-weight:800;color:#f8fafc;margin-bottom:12px;">
                                        🏟️ {fix['HomeTeam']} <span style="color:#64748b;font-weight:400;">vs</span> ✈️ {fix['AwayTeam']}
                                    </div>
                                    <div style="display:flex;justify-content:space-between;font-size:0.88rem;margin-bottom:6px;">
                                        <span><strong style="color:#3b82f6;">Home:</strong> {fix['prob_home']*100:.1f}%</span>
                                        <span><strong style="color:#f59e0b;">Draw:</strong> {fix['prob_draw']*100:.1f}%</span>
                                        <span><strong style="color:#ef4444;">Away:</strong> {fix['prob_away']*100:.1f}%</span>
                                    </div>
                                    <div style="width:100%;height:10px;border-radius:99px;background:#1e293b;display:flex;overflow:hidden;margin-bottom:10px;">
                                        <div style="width:{fix['prob_home']*100:.1f}%;background:#3b82f6;"></div>
                                        <div style="width:{fix['prob_draw']*100:.1f}%;background:#f59e0b;"></div>
                                        <div style="width:{fix['prob_away']*100:.1f}%;background:#ef4444;"></div>
                                    </div>
                                    <div style="font-size:0.78rem;color:#94a3b8;border-top:1px solid #334155;padding-top:6px;display:flex;justify-content:space-between;">
                                        <span>Elo: {fix['home_elo']:.0f} vs {fix['away_elo']:.0f} ({fix['elo_diff']:+.0f})</span>
                                        <span>Form Diff: {fix['Delta_Points_Roll_5']:+.2f} pts</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )


if __name__ == "__main__":
    main()
