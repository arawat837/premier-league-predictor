"""End-to-end execution script orchestrating ingestion, feature engineering, training, and evaluation."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data_loader import get_data
from src.features import build_and_save_features
from src.models import train_and_evaluate_all


def run_pipeline(force_download: bool = False) -> dict:
    """Run the complete Premier League Match Outcome & Probability Predictor pipeline."""
    raw_dir = Path("data") / "raw"
    features_path = Path("data") / "processed" / "features.parquet"
    model_path = Path("models") / "calibrated_xgb.joblib"

    get_data(raw_dir=raw_dir, force_download=force_download)
    build_and_save_features(raw_dir=raw_dir, output_path=features_path)
    results = train_and_evaluate_all(features_path=features_path, model_output_path=model_path)
    return results


if __name__ == "__main__":
    run_pipeline()
