"""Evaluation metrics, confusion matrix plotting, and high-conviction filtering.

Provides:
- Multiclass Log Loss and Multiclass Brier Score calculation
- Confusion Matrix computation, console table rendering, and image export
- High-Conviction Filter evaluating precision/hit-rate at max(P) >= 0.65
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from PIL import Image, ImageDraw, ImageFont

CLASS_LABELS = {0: "Away Win (0)", 1: "Draw (1)", 2: "Home Win (2)"}
SHORT_LABELS = ["Away (0)", "Draw (1)", "Home (2)"]


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 3) -> float:
    """Compute the multiclass Brier score.

    Formula: (1 / N) * sum_{i=1..N} sum_{c=0..C-1} (p_{i,c} - y_{i,c})^2
    where y_{i,c} is the one-hot indicator for the true outcome.

    Args:
        y_true: Array of integer class labels (0, 1, 2).
        y_prob: Array of predicted probabilities of shape (N, 3).
        num_classes: Number of target classes (3).

    Returns:
        Multiclass Brier score (float).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    one_hot = np.eye(num_classes)[y_true_arr]
    return float(np.mean(np.sum((y_prob_arr - one_hot) ** 2, axis=1)))


def evaluate_high_conviction(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.65,
) -> Dict[str, Any]:
    """Evaluate predictions specifically on high-conviction matches where max(P) >= threshold.

    Args:
        y_true: Array of true class integers (0, 1, 2).
        y_prob: Array of calibrated class probabilities of shape (N, 3).
        threshold: Minimum probability threshold to qualify a pick (default 0.65).

    Returns:
        Dictionary with count, coverage, accuracy, and per-threshold breakdown.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    max_probs = np.max(y_prob_arr, axis=1)
    y_pred = np.argmax(y_prob_arr, axis=1)

    mask = max_probs >= threshold
    total_matches = len(y_true_arr)
    qualified_count = int(np.sum(mask))
    coverage = (qualified_count / total_matches) if total_matches > 0 else 0.0

    if qualified_count > 0:
        hc_accuracy = float(accuracy_score(y_true_arr[mask], y_pred[mask]))
        hc_brier = multiclass_brier_score(y_true_arr[mask], y_prob_arr[mask])
        hc_logloss = float(log_loss(y_true_arr[mask], y_prob_arr[mask], labels=[0, 1, 2]))
    else:
        hc_accuracy = float("nan")
        hc_brier = float("nan")
        hc_logloss = float("nan")

    # Also compute threshold sweep curve
    sweep_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    threshold_curve: List[Dict[str, Any]] = []
    for th in sweep_thresholds:
        th_mask = max_probs >= th
        th_count = int(np.sum(th_mask))
        th_acc = float(accuracy_score(y_true_arr[th_mask], y_pred[th_mask])) if th_count > 0 else float("nan")
        threshold_curve.append({
            "threshold": th,
            "matches": th_count,
            "coverage_pct": (th_count / total_matches * 100.0) if total_matches > 0 else 0.0,
            "accuracy": th_acc,
        })

    return {
        "threshold": threshold,
        "total_matches": total_matches,
        "qualified_matches": qualified_count,
        "coverage": coverage,
        "accuracy": hc_accuracy,
        "log_loss": hc_logloss,
        "brier_score": hc_brier,
        "threshold_curve": threshold_curve,
    }


def format_confusion_matrix_text(cm: np.ndarray, title: str = "Confusion Matrix") -> str:
    """Format a 3x3 confusion matrix as a clean ASCII table for console display."""
    header_label = "Actual / Predicted"
    lines = [
        f"\n{title}",
        "-" * 62,
        f"{header_label:<20} | {'Away (0)':>10} | {'Draw (1)':>10} | {'Home (2)':>10} | {'Total':>6}",
        "-" * 62,
    ]
    for i, label in enumerate(SHORT_LABELS):
        row_sum = int(np.sum(cm[i]))
        lines.append(
            f"{label:<20} | {cm[i, 0]:>10d} | {cm[i, 1]:>10d} | {cm[i, 2]:>10d} | {row_sum:>6d}"
        )
    lines.append("-" * 62)
    col_sums = np.sum(cm, axis=0)
    total_all = int(np.sum(cm))
    lines.append(
        f"{'Predicted Total':<20} | {col_sums[0]:>10d} | {col_sums[1]:>10d} | {col_sums[2]:>10d} | {total_all:>6d}"
    )
    lines.append("-" * 62)
    return "\n".join(lines)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Out-of-Fold Confusion Matrix (Calibrated XGBoost)",
) -> np.ndarray:
    """Compute, print, and save a visual plot of the 3x3 confusion matrix.

    Args:
        y_true: True integer class labels (0, 1, 2).
        y_pred: Predicted integer class labels (0, 1, 2).
        output_path: Optional file path to save the visual PNG heatmap.
        title: Plot title.

    Returns:
        3x3 numpy confusion matrix array.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Print formatted console confusion matrix
    print(format_confusion_matrix_text(cm, title=title))

    if output_path is None:
        output_path = Path("models") / "confusion_matrix.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Render heatmap PNG using Pillow
    width, height = 720, 620
    img = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default()

    # Title
    draw.text((140, 25), title, fill=(15, 23, 42), font=font)
    draw.text((250, 50), "Predicted Outcome", fill=(51, 65, 85), font=font)

    grid_x0, grid_y0 = 180, 110
    cell_w, cell_h = 140, 120
    max_val = max(int(cm.max()), 1)

    # Column headers
    for col_idx, col_label in enumerate(SHORT_LABELS):
        cx = grid_x0 + col_idx * cell_w + 35
        draw.text((cx, grid_y0 - 25), col_label, fill=(30, 41, 59), font=font)

    # Rows and cells
    for row_idx, row_label in enumerate(SHORT_LABELS):
        ry = grid_y0 + row_idx * cell_h + 50
        draw.text((55, ry), f"Actual {row_label}", fill=(30, 41, 59), font=font)

        row_total = max(int(cm[row_idx].sum()), 1)
        for col_idx in range(3):
            val = int(cm[row_idx, col_idx])
            pct = (val / row_total) * 100.0
            intensity = val / max_val

            # Blue gradient background
            r = int(235 - 190 * intensity)
            g = int(243 - 140 * intensity)
            b = int(255 - 45 * intensity)
            text_color = (255, 255, 255) if intensity > 0.55 else (15, 23, 42)

            x0 = grid_x0 + col_idx * cell_w
            y0 = grid_y0 + row_idx * cell_h
            x1 = x0 + cell_w - 4
            y1 = y0 + cell_h - 4

            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b), outline=(203, 213, 225), width=2)
            draw.text((x0 + 48, y0 + 42), f"{val}", fill=text_color, font=font)
            draw.text((x0 + 38, y0 + 65), f"({pct:.1f}%)", fill=text_color, font=font)

    img.save(output_path)
    print(f"Saved confusion matrix plot to {output_path}")
    return cm


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.65,
    save_plot_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Full evaluation suite computing Log Loss, Brier Score, Confusion Matrix, and High-Conviction accuracy.

    Args:
        y_true: Ground-truth class labels (0, 1, 2).
        y_prob: Predicted class probabilities (N, 3).
        model_name: Display name of model being evaluated.
        threshold: Probability threshold for high-conviction picks (default 0.65).
        save_plot_path: Optional path to save confusion matrix PNG.

    Returns:
        Dictionary of evaluation metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = np.argmax(y_prob_arr, axis=1)

    ll = float(log_loss(y_true_arr, y_prob_arr, labels=[0, 1, 2]))
    bs = multiclass_brier_score(y_true_arr, y_prob_arr)
    acc = float(accuracy_score(y_true_arr, y_pred))

    hc = evaluate_high_conviction(y_true_arr, y_prob_arr, threshold=threshold)

    cm = plot_confusion_matrix(
        y_true_arr,
        y_pred,
        output_path=save_plot_path,
        title=f"Confusion Matrix - {model_name} (All Out-of-Fold Matches)",
    )

    return {
        "model_name": model_name,
        "log_loss": ll,
        "brier_score": bs,
        "accuracy": acc,
        "confusion_matrix": cm,
        "high_conviction": hc,
    }
