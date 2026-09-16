"""
Chương trình đánh giá toàn diện End-to-End và tách biệt từng giai đoạn (A, B, C, FEN).
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .metrics_stage_a import compute_corner_l2_errors, compute_homography_reprojection_error
from .metrics_stage_b import compute_piece_detection_metrics
from .metrics_stage_c import compute_classification_metrics
from ..post_processing.piece_definitions import BOARD_NUM_ROWS, BOARD_NUM_COLS


def evaluate_end_to_end_board_state(
    pred_board_matrix: List[List[Optional[int]]],  # 10x9 class_id hoặc None
    gt_board_matrix: List[List[Optional[int]]],
) -> Dict[str, float]:
    """
    Đo đạc độ chính xác ma trận bàn cờ 90 giao điểm và so sánh FEN.
    """
    total_intersections = BOARD_NUM_ROWS * BOARD_NUM_COLS  # 90
    exact_matches = 0
    piece_pos_matches = 0
    total_gt_pieces = 0

    for r in range(BOARD_NUM_ROWS):
        for c in range(BOARD_NUM_COLS):
            p = pred_board_matrix[r][c]
            g = gt_board_matrix[r][c]

            if p == g:
                exact_matches += 1

            if g is not None:
                total_gt_pieces += 1
                if p == g:
                    piece_pos_matches += 1

    board_accuracy = exact_matches / total_intersections
    piece_accuracy = piece_pos_matches / max(total_gt_pieces, 1)
    is_perfect_match = 1.0 if exact_matches == total_intersections else 0.0

    return {
        "board_intersection_accuracy": float(board_accuracy),
        "piece_exact_accuracy": float(piece_accuracy),
        "is_perfect_match": float(is_perfect_match),
    }


def print_evaluation_summary_report(
    stage_a_metrics: Optional[Dict[str, float]] = None,
    stage_b_metrics: Optional[Dict[str, float]] = None,
    stage_c_metrics: Optional[Dict[str, float]] = None,
    e2e_metrics: Optional[Dict[str, float]] = None,
):
    """In bảng tổng kết đánh giá chi tiết theo từng giai đoạn ra màn hình"""
    print("\n" + "=" * 65)
    print("      BÁO CÁO ĐÁNH GIÁ HỆ THỐNG THỊ GIÁC CỜ TƯỚNG (XIANGQI CV)      ")
    print("=" * 65)

    if stage_a_metrics:
        print("\n[ GIAI ĐOẠN A: ĐỊNH VỊ BÀN CỜ & HOMOGRAPHY ]")
        print(f" - Sai số góc bàn trung bình (MAE): {stage_a_metrics.get('mean_corner_error_px', 0):.2f} px")
        print(f" - Sai số góc bàn RMSE:           {stage_a_metrics.get('rmse_corner_error_px', 0):.2f} px")
        print(f" - Sai số lớn nhất (Max Error):     {stage_a_metrics.get('max_corner_error_px', 0):.2f} px")

    if stage_b_metrics:
        print("\n[ GIAI ĐOẠN B: ĐỊNH VỊ QUÂN CỜ (DETECTION) ]")
        print(f" - Precision:                     {stage_b_metrics.get('precision', 0) * 100:.2f}%")
        print(f" - Recall:                        {stage_b_metrics.get('recall', 0) * 100:.2f}%")
        print(f" - F1-Score:                      {stage_b_metrics.get('f1_score', 0):.4f}")
        print(f" - Tỷ lệ sót quân (Miss Rate):    {stage_b_metrics.get('miss_rate', 0) * 100:.2f}%")
        print(f" - Tỷ lệ quân ma (Ghost Rate):    {stage_b_metrics.get('ghost_rate', 0) * 100:.2f}%")
        print(f" - Sai số lệch tâm trung bình:    {stage_b_metrics.get('mean_loc_error_px', 0):.2f} px")

    if stage_c_metrics:
        print("\n[ GIAI ĐOẠN C: PHÂN LOẠI 14 LOẠI QUÂN CỜ ]")
        print(f" - Top-1 Accuracy:                {stage_c_metrics.get('top1_accuracy', 0) * 100:.2f}%")
        print(f" - Macro F1-Score:                {stage_c_metrics.get('macro_f1', 0):.4f}")
        print(f" - Weighted F1-Score:             {stage_c_metrics.get('weighted_f1', 0):.4f}")
        print(f" - Tỷ lệ nhầm màu (Red vs Black): {stage_c_metrics.get('color_confusion_rate', 0) * 100:.2f}%")

    if e2e_metrics:
        print("\n[ KẾT QUẢ TOÀN DIỆN END-TO-END ]")
        print(f" - Độ chính xác 90 giao điểm:     {e2e_metrics.get('board_intersection_accuracy', 0) * 100:.2f}%")
        print(f" - Độ chính xác nhận diện quân:   {e2e_metrics.get('piece_exact_accuracy', 0) * 100:.2f}%")
        print(f" - Tỷ lệ ván khớp 100% (Exact FEN):{e2e_metrics.get('is_perfect_match', 0) * 100:.2f}%")

    print("=" * 65 + "\n")
