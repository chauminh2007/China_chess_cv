"""
Đánh giá độc lập Giai đoạn C: Độ chính xác phân loại 14 loại quân cờ và Confusion Matrix.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from ..post_processing.piece_definitions import PIECE_REGISTRY


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[np.ndarray] = None,
) -> Dict[str, any]:
    """
    Tính toán các chỉ số phân loại chi tiết: Top-1, Macro-F1, Weighted-F1, Confusion Matrix.
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)

    top1_acc = float(accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    top2_acc = 0.0
    if y_probs is not None and y_probs.shape[1] >= 2:
        try:
            top2_acc = float(top_k_accuracy_score(y_t, y_probs, k=2, labels=list(range(14))))
        except Exception:
            pass

    # Tính Confusion Matrix 14x14
    cm = confusion_matrix(y_t, y_p, labels=list(range(14)))

    # Phân tích tỷ lệ nhầm lẫn màu quân (Red vs Black)
    # Red: 0..6, Black: 7..13
    red_true_mask = y_t < 7
    black_true_mask = y_t >= 7

    red_as_black = np.sum((y_p[red_true_mask] >= 7)) if np.sum(red_true_mask) > 0 else 0
    black_as_red = np.sum((y_p[black_true_mask] < 7)) if np.sum(black_true_mask) > 0 else 0
    total_samples = max(len(y_t), 1)
    color_confusion_rate = float((red_as_black + black_as_red) / total_samples)

    return {
        "top1_accuracy": top1_acc,
        "top2_accuracy": top2_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "color_confusion_rate": color_confusion_rate,
        "confusion_matrix": cm,
    }
