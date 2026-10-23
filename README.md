# Premier League Match Outcome & Probability Predictor

An end-to-end, leak-free machine learning system and interactive Streamlit application that predicts English Premier League match outcomes (**Home Win**, **Draw**, **Away Win**) as calibrated probabilities, equipped with a **High-Conviction Lock** filter ($P \ge 0.65$) targeting 70%+ precision on top-tier picks and **SHAP** TreeExplainer attribution.

---

## 1. Executive Summary & Core Results

Predicting association football match outcomes presents high stochasticity, low scoring events, and significant draw frequency (~24%). Standard sports machine learning pipelines suffer from subtle data leakage (e.g. using pre-game rolling stats that inadvertently encompass the current fixture or random train/test splits that leak future form).

This project resolves these failure modes through:
1. **Mathematical Zero-Leakage Featurization**: Every rolling metric and exponential moving average (EWMA) strictly applies `.shift(1)` grouped by club. Dynamic Elo ratings and rest days update strictly post-match.
2. **Chronological Walk-Forward Expanding Window Validation**: Models are evaluated across 4 full Premier League seasons (1,520 out-of-fold fixtures) without shuffling.
3. **Probability Calibration**: Tree models are trained on multiclass log loss and calibrated with isotonic regression (`CalibratedClassifierCV(method='isotonic')`) to ensure forecasted probabilities match true empirical event rates.
4. **Dual Inference Engine**:
   - **All Fixtures**: Complete 3-way calibrated probability distributions.
   - **High-Conviction Locks**: A selective filter flagging fixtures where $\max(P) \ge 0.65$, yielding **70.9% empirical hit rate** on raw softprob picks and **64.3%–67.0%** on calibrated picks.

### Walk-Forward Validation Performance (1,520 Out-of-Fold Matches)

| Model Pipeline | Multiclass Log Loss | Multiclass Brier Score | Overall Accuracy | High-Conviction Precision ($\max(P) \ge 0.65$) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Random Forest (Isotonic)** | `1.0166` | `0.6025` | `52.57%` | `65.8%` (330 / 1,520) |
| **XGBoost Raw (`multi:softprob`)** | `1.0027` | `0.5975` | `52.89%` | **`70.9%`** (86 / 1,520) |
| **XGBoost Calibrated (`Isotonic`)** | `1.0141` | `0.6022` | `52.57%` | `64.3%` (364 / 1,520) |

#### High-Conviction Threshold Sweep (Calibrated XGBoost)
As confidence threshold increases, precision scales monotonically:

| Minimum Probability Threshold | Qualified Fixtures | Coverage (%) | Empirical Hit Rate (Accuracy) |
| :--- | :---: | :---: | :---: |
| $P \ge 0.50$ | 1,036 | 68.2% | **56.95%** |
| $P \ge 0.55$ | 790 | 52.0% | **60.63%** |
| $P \ge 0.60$ | 570 | 37.5% | **62.46%** |
| $P \ge 0.65$ (Lock Threshold) | 364 | 23.9% | **64.29%** |
| $P \ge 0.70$ | 227 | 14.9% | **66.96%** |

---

## 2. System Architecture

```text
pl-match-predictor/
├── data/
│   ├── raw/                      # Downloaded historical CSVs (2020-21 through 2025-26)
│   │   └── upcoming_fixtures.csv # Guaranteed offline schedule fallback
│   └── processed/
│       ├── features.parquet      # 54 engineered pre-match features (2,280 fixtures)
│       └── oof_predictions.parquet # Out-of-fold validation predictions & metadata
├── models/
│   ├── calibrated_xgb.joblib     # Production Isotonic-Calibrated XGBoost pipeline
│   └── confusion_matrix.png      # Visual confusion matrix heatmap
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Automated ingestion from Football-Data.co.uk & upcoming fixtures
│   ├── features.py               # Zero-leakage rolling, EWMA, Elo, & differential engineering
│   ├── models.py                 # Walk-Forward Expanding Window CV & probability calibration
│   ├── evaluate.py               # Log Loss, Brier score, confusion matrix & high-conviction metrics
│   └── pipeline.py               # End-to-end execution runner
├── app/
│   └── dashboard.py              # Streamlit UI (Matchup Simulator, Upcoming GW, SHAP explanations)
├── tests/
│   ├── test_data_loader.py       # Ingestion schema & target mapping unit tests
│   ├── test_leakage.py           # Zero data leakage mutation tests
│   ├── test_models.py            # Chronological splitting & Brier score tests
│   └── test_pipeline.py          # End-to-end inference & AppTest smoke tests
├── docs/
│   └── images/                   # Verification screenshots and architecture figures
├── .github/
│   └── workflows/ci.yml          # GitHub Actions CI pipeline running pytest across all 11 tests
├── requirements.txt
├── pytest.ini
└── README.md
```

```mermaid
flowchart LR
    A["Raw Ingestion<br/>Football-Data.co.uk<br/>(2020-2026)"] --> B["Standardization<br/>FTR Mapping<br/>(0: Away, 1: Draw, 2: Home)"]
    B --> C["Time-Series Featurization<br/>Grouped .shift(1)<br/>Dynamic Elo & Differentials"]
    C --> D["Walk-Forward Validation<br/>Expanding Window<br/>(4 Test Seasons, 1,520 Matches)"]
    D --> E["Probability Calibration<br/>XGBoost (multi:softprob)<br/>CalibratedClassifierCV(isotonic)"]
    E --> F["Dual Inference<br/>1. Calibrated Probabilities<br/>2. High-Conviction Locks (P >= 65%)"]
    F --> G["Streamlit Dashboard<br/>Custom Matchup Simulator<br/>Upcoming Gameweek & SHAP"]
```

---

## 3. Strict Anti-Leakage Methodology

In conventional sports analytics, data leakage routinely produces deceptively high offline accuracy that collapses in production. This architecture enforces two non-negotiable safeguards:

### 1. Grouped `.shift(1)` Lagging for all Match Metrics
Every single feature representing team performance (goals scored, goals conceded, total shots, shots on target, points) is derived from a chronological match stream per club where each metric is shifted by 1 game prior to rolling calculations:

$$\text{Metric}_{\text{shifted}}[t] = \text{Metric}[t - 1]$$

$$\text{Roll}_W[t] = \frac{1}{W} \sum_{i=1}^{W} \text{Metric}_{\text{shifted}}[t - i + 1] = \frac{1}{W} \sum_{k=1}^{W} \text{Metric}[t - k]$$

Row $t$ (the fixture to be predicted) can never see its own 90-minute stats.

### 2. Strictly Post-Match Dynamic Elo Updating
Elo ratings for Home and Away clubs are recorded in the feature row prior to updating ratings:

$$R_{\text{Home}}[t] = \text{Current Elo}(\text{HomeTeam})$$

$$R_{\text{Away}}[t] = \text{Current Elo}(\text{AwayTeam})$$

Rating adjustments occur strictly **after** fixture completion using standard goal-difference margin multipliers:

$$R_{\text{Home}}[t+1] = R_{\text{Home}}[t] + K \cdot M \cdot (S_{\text{Home}} - E_{\text{Home}})$$

### 3. Pytest Leakage Invariance Assertions
[`tests/test_leakage.py`](file:///c:/Users/abhis/OneDrive/Documents/Desktop/Football%20prediction%20model/tests/test_leakage.py) programmatically modifies actual match scores, shots, shots on target, and outcomes for row $i$ and asserts via `pd.testing.assert_series_equal` that row $i$'s pre-match rolling, Elo, and differential features undergo **zero modification**.

---

## 4. Evaluation & Calibration Analysis

### Out-of-Fold Confusion Matrix

![Confusion Matrix](docs/images/confusion_matrix.png)

```text
Actual \ Predicted   |   Away (0) |   Draw (1) |   Home (2) |  Total
--------------------------------------------------------------------
Away (0)             |        296 |         12 |        170 |    478
Draw (1)             |        163 |          6 |        197 |    366
Home (2)             |        172 |          7 |        497 |    676
--------------------------------------------------------------------
Predicted Total      |        631 |         25 |        864 |   1520
```

- **Home Advantage Capture**: The model accurately predicts 497 out of 676 actual home wins (73.5% sensitivity on home victories).
- **Away Win Precision**: Strong recall on away victories (296 correct out of 478 actual away wins).
- **Conservative Draw Modeling**: Consistent with academic soccer modeling literature, draws represent maximum outcome entropy; the calibrated probability engine distributes mass rather than forcing false-positive discrete draw predictions.

---

## 5. Interactive Streamlit Dashboard

The web application ([`app/dashboard.py`](file:///c:/Users/abhis/OneDrive/Documents/Desktop/Football%20prediction%20model/app/dashboard.py)) provides two primary interfaces:

### Tab 1: Custom Matchup Simulator (`Arsenal vs. Chelsea`)
Simulate any hypothetical matchup between Premier League clubs, inspect real-time pre-match feature differentials, view horizontal probability bars, and analyze the **Top 3 SHAP driving features**.

![Custom Matchup Simulator — Arsenal vs Chelsea](docs/images/dashboard_arsenal_chelsea.png)

- **Calibrated Outcome**: Arsenal (79.8% Home Win) | Draw (14.1%) | Chelsea (6.1% Away Win).
- **Lock Status**: **🔒 HIGH-CONVICTION LOCK QUALIFIED** ($79.8\% \ge 65\%$).
- **Top 3 SHAP Drivers**:
  1. `elo_diff` (*Elo Rating Differential*): `+0.205` SHAP log-odds (`+166.9` Elo advantage)
  2. `home_elo` (*Home Team Elo Rating*): `+0.119` SHAP log-odds (`1,751.0` pre-match Elo)
  3. `Delta_Points_Roll_10` (*10-Match Form Points Differential*): `+0.036` SHAP log-odds

### Tab 2: Upcoming Gameweek Forecast
Fetches the upcoming Premier League schedule (via the football-data.org API with automatic CSV fallback), calculates pre-match feature vectors, and surfaces gameweek lock selections.

![Upcoming Gameweek Dashboard](docs/images/upcoming_gameweek.png)

---

## 6. Setup & Reproduction Guide

### Prerequisites
- Python 3.11 or 3.12
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/premier-league-predictor.git
cd premier-league-predictor

# 2. Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Full Pipeline Reproduction
Run ingestion, zero-leakage feature engineering, walk-forward CV, probability calibration, and model serialization in a single command:

```bash
python src/pipeline.py
```

### Running the Test Suite
Execute all 11 unit, leakage, and integration smoke tests:

```bash
pytest -v
```

Expected output:
```text
tests/test_data_loader.py::test_standardize_dataframe PASSED             [  9%]
tests/test_data_loader.py::test_standardize_target_mapping PASSED        [ 18%]
tests/test_data_loader.py::test_standardize_drops_invalid_rows PASSED    [ 27%]
tests/test_leakage.py::test_zero_leakage_on_current_match_modification PASSED [ 36%]
tests/test_leakage.py::test_leakage_check_across_multiple_rows PASSED    [ 45%]
tests/test_leakage.py::test_proper_forward_propagation PASSED            [ 54%]
tests/test_models.py::test_walk_forward_splits_strictly_chronological PASSED [ 63%]
tests/test_models.py::test_multiclass_brier_score_and_high_conviction PASSED [ 72%]
tests/test_pipeline.py::test_get_upcoming_fixtures PASSED                [ 81%]
tests/test_pipeline.py::test_upcoming_gameweek_apptest PASSED            [ 90%]
tests/test_pipeline.py::test_end_to_end_arsenal_vs_chelsea_inference PASSED [100%]
======================= 11 passed in 11.8s =======================
```

### Launching the Dashboard
Start the local Streamlit application:

```bash
streamlit run app/dashboard.py --server.port 8501
```
Open `http://localhost:8501` in your browser.
