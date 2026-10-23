# Agent Task Specification: Premier League Match Outcome & Probability Predictor

## 1. Project Goal
Build an end-to-end, leak-free machine learning system and interactive dashboard that predicts English Premier League match outcomes (Home Win, Draw, Away Win) as calibrated probabilities, with a high-conviction filtering system targeting 70%+ precision on top-tier picks.

---

## 2. Strict Technical Constraints & Anti-Leakage Rules (Non-Negotiables)
1. **Zero Data Leakage:**
   - Pre-match team performance metrics (goals scored, goals conceded, shots, shots on target, possession, corners) must strictly use `.shift(1)` grouped by team before applying rolling or exponentially weighted windows.
   - Elo ratings and points tallies must be updated strictly *after* the current fixture result is evaluated.
2. **Chronological Validation:**
   - Never use random `train_test_split` or standard k-fold cross-validation.
   - Use Walk-Forward Expanding Window validation: train on historical seasons (e.g., 2018–2024), evaluate on subsequent fixtures (e.g., 2024–2026), and advance iteratively.
3. **Probability Calibration:**
   - Train models using multiclass log loss / multi-softprob objectives.
   - Wrap the best-performing tree models (XGBoost / LightGBM / CatBoost) with isotonic or Platt probability calibration (`CalibratedClassifierCV`) to ensure predicted probabilities mirror true frequencies.
4. **Dual Prediction Output:**
   - **All Matches:** Calibrated probabilities across all 3 classes (e.g., `Home: 52%`, `Draw: 26%`, `Away: 22%`).
   - **High-Conviction Picks:** A filtering system that tags fixtures where the maximum win probability exceeds a tunable threshold (e.g., $P \ge 0.65$), tracking accuracy and hit rate specifically on these qualified selections.

---

## 3. Target Project Architecture

```text
pl-match-predictor/
├── data/
│   ├── raw/                 # Downloaded historical CSVs (Football-Data / FBref)
│   └── processed/           # Engineered feature tables
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Ingests and standardizes match data across seasons
│   ├── features.py          # Lagged rolling metrics, EWMA, differential features, Elo
│   ├── models.py            # Model training, hyperparameter search, calibration
│   ├── evaluate.py          # Log loss, Brier score, confusion matrix, threshold curves
│   └── pipeline.py          # End-to-end execution script
├── app/
│   └── dashboard.py         # Streamlit interactive UI with probability bars & SHAP
├── tests/
│   ├── test_leakage.py      # Unit tests ensuring no future stats leak into past rows
│   └── test_pipeline.py     # End-to-end smoke tests
├── requirements.txt
└── README.md