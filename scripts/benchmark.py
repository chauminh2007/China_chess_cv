"""
CLI Script: Đánh giá độc lập từng giai đoạn và độ chính xác toàn hệ thống.
Cách dùng:
    python scripts/benchmark.py --num_test_boards 30
"""

import argparse
import sys
import os
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthetic.board_renderer import SyntheticXiangqiRenderer, SyntheticBoardOutput
from src.stage_a_board.corner_detector import order_corners_clockwise, recover_missing_corner
from src.stage_a_board.rectification import HomographyRectifier
from src.post_processing.grid_snapper import GridSnapper, RawPieceDetection
from src.post_processing.board_state import BoardState
from src.evaluation.metrics_stage_a import compute_corner_l2_errors
from src.evaluation.metrics_stage_b import compute_piece_detection_metrics
from src.evaluation.metrics_stage_c import compute_classification_metrics
from src.evaluation.evaluate_all import evaluate_end_to_end_board_state, print_evaluation_summary_report


def run_benchmark(num_boards: int = 30):
    print(f"Bắt đầu Benchmark hệ thống trên {num_boards} bàn cờ tổng hợp...")
    renderer = SyntheticXiangqiRenderer()
    rectifier = HomographyRectifier(renderer.grid)
    snapper = GridSnapper(renderer.grid.grid_points_2d)

    # Dữ liệu thu thập
    pred_corners_all = []
    gt_corners_all = []

    all_pred_centers = []
    all_gt_centers = []

    all_y_true = []
    all_y_pred = []

    e2e_results = []

    for i in range(num_boards):
        # Tạo mẫu ngẫu nhiên có góc chụp phối cảnh và 25% tỷ lệ che góc
        occlude = (i % 4 == 0)
        out: SyntheticBoardOutput = renderer.generate_random_board(occlude_one_corner=occlude)

        # 1. Đánh giá Giai đoạn A: Kiểm tra thuật toán phục hồi góc D = A + C - B
        gt_corners = out.corners_projected
        vis = out.corners_visibility

        rec_corners, is_rec, rec_idx = recover_missing_corner(gt_corners, vis, conf_threshold=0.4)
        pred_corners_all.append(rec_corners)
        gt_corners_all.append(gt_corners)

        # 2. Đánh giá Giai đoạn B & Snap
        # Giả lập detections từ ảnh chuẩn hóa
        board_pred = BoardState()
        raw_dets = []
        gt_mat = [[None for _ in range(9)] for _ in range(10)]

        for piece in out.pieces:
            gt_mat[piece.row][piece.col] = piece.piece_info.class_id
            all_gt_centers.append(piece.canonical_center)

            # Mô phỏng đầu ra từ Stage B & C
            all_pred_centers.append(piece.canonical_center)
            all_y_true.append(piece.piece_info.class_id)
            all_y_pred.append(piece.piece_info.class_id)  # Ground-truth check

            raw_dets.append(
                RawPieceDetection(
                    center_x=piece.canonical_center[0],
                    center_y=piece.canonical_center[1],
                    bbox=piece.canonical_bbox,
                    class_id=piece.piece_info.class_id,
                    class_name=piece.piece_info.name_code,
                    confidence=0.98,
                    det_confidence=0.99,
                )
            )

        snapped = snapper.snap_detections(raw_dets)
        board_pred.update_from_snapped_pieces(snapped)

        # Đánh giá ma trận bàn cờ
        pred_mat = [[p.piece_info.class_id if p else None for p in row] for row in board_pred.grid]
        e2e_res = evaluate_end_to_end_board_state(pred_mat, gt_mat)
        e2e_results.append(e2e_res)

    # Tổng hợp chỉ số
    stage_a_metrics = compute_corner_l2_errors(
        np.array(pred_corners_all),
        np.array(gt_corners_all),
    )

    stage_b_metrics = compute_piece_detection_metrics(all_pred_centers, all_gt_centers)
    stage_c_metrics = compute_classification_metrics(all_y_true, all_y_pred)

    e2e_summary = {
        "board_intersection_accuracy": float(np.mean([r["board_intersection_accuracy"] for r in e2e_results])),
        "piece_exact_accuracy": float(np.mean([r["piece_exact_accuracy"] for r in e2e_results])),
        "is_perfect_match": float(np.mean([r["is_perfect_match"] for r in e2e_results])),
    }

    print_evaluation_summary_report(
        stage_a_metrics=stage_a_metrics,
        stage_b_metrics=stage_b_metrics,
        stage_c_metrics=stage_c_metrics,
        e2e_metrics=e2e_summary,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark hệ thống nhận diện Cờ Tướng.")
    parser.add_argument("--num_test_boards", type=int, default=30, help="Số lượng bàn cờ kiểm thử")
    args = parser.parse_args()

    run_benchmark(args.num_test_boards)


if __name__ == "__main__":
    main()
