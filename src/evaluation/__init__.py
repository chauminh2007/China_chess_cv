"""
Package Đánh giá & Benchmark từng giai đoạn độc lập (Stage Evaluation & Metrics).
"""

from .metrics_stage_a import compute_corner_l2_errors, compute_homography_reprojection_error
from .metrics_stage_b import compute_piece_detection_metrics
from .metrics_stage_c import compute_classification_metrics
from .evaluate_all import evaluate_end_to_end_board_state, print_evaluation_summary_report

__all__ = [
    "compute_corner_l2_errors",
    "compute_homography_reprojection_error",
    "compute_piece_detection_metrics",
    "compute_classification_metrics",
    "evaluate_end_to_end_board_state",
    "print_evaluation_summary_report",
]
