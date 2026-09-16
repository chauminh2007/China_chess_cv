"""
Unit tests kiểm tra Hậu xử lý: Snap giao điểm lưới, Ràng buộc luật cờ và Chuỗi FEN.
"""

import unittest
import numpy as np

from src.stage_a_board.canonical_grid import CanonicalBoardGrid
from src.post_processing.grid_snapper import GridSnapper, RawPieceDetection
from src.post_processing.board_state import BoardState
from src.post_processing.piece_definitions import (
    PIECE_REGISTRY,
    NAME_TO_CLASS_ID,
    PieceColor,
    PieceType,
)


class TestPostProcessing(unittest.TestCase):

    def setUp(self):
        self.grid = CanonicalBoardGrid(canvas_width=720, canvas_height=800, margin_x=40, margin_y=40)
        self.snapper = GridSnapper(self.grid.grid_points_2d, max_snap_radius_px=38.0)

    def test_grid_snapping_exact_and_jittered(self):
        """Kiểm tra snap giao điểm với tọa độ có nhiễu nhẹ (jitter)"""
        # Giao điểm (cột 4, hàng 0) -> Vị trí Tướng Đen ban đầu
        base_x, base_y = self.grid.get_pixel_coord(4, 0)

        # Thêm nhiễu 5px
        jitter_x = base_x + 5.0
        jitter_y = base_y - 4.0

        det = RawPieceDetection(
            center_x=jitter_x,
            center_y=jitter_y,
            bbox=(jitter_x - 20, jitter_y - 20, jitter_x + 20, jitter_y + 20),
            class_id=NAME_TO_CLASS_ID["b_king"],
            class_name="b_king",
            confidence=0.95,
        )

        snapped = self.snapper.snap_detections([det])
        self.assertEqual(len(snapped), 1)
        self.assertEqual(snapped[0].col, 4)
        self.assertEqual(snapped[0].row, 0)
        self.assertEqual(snapped[0].piece_info.piece_type, PieceType.KING)
        self.assertEqual(snapped[0].piece_info.color, PieceColor.BLACK)

    def test_out_of_bounds_filtering(self):
        """Kiểm tra lọc bỏ quân nằm ngoài bàn cờ hoặc lệch quá xa giao điểm"""
        det_outside = RawPieceDetection(
            center_x=10.0,  # Nằm sát mép ngoài (lề là 40)
            center_y=10.0,
            bbox=(0, 0, 20, 20),
            class_id=NAME_TO_CLASS_ID["r_soldier"],
            class_name="r_soldier",
            confidence=0.88,
        )

        snapped = self.snapper.snap_detections([det_outside])
        self.assertEqual(len(snapped), 0, "Quân ngoài bàn cờ phải bị lọc bỏ")

    def test_conflict_resolution_highest_conf(self):
        """Kiểm tra giải quyết xung đột khi 2 detection snap vào cùng 1 ô"""
        base_x, base_y = self.grid.get_pixel_coord(0, 0)

        det_1 = RawPieceDetection(
            center_x=base_x + 2.0,
            center_y=base_y + 1.0,
            bbox=(base_x - 15, base_y - 15, base_x + 15, base_y + 15),
            class_id=NAME_TO_CLASS_ID["b_chariot"],
            class_name="b_chariot",
            confidence=0.98,
        )

        det_2 = RawPieceDetection(
            center_x=base_x + 4.0,
            center_y=base_y + 3.0,
            bbox=(base_x - 15, base_y - 15, base_x + 15, base_y + 15),
            class_id=NAME_TO_CLASS_ID["b_soldier"],
            class_name="b_soldier",
            confidence=0.45,  # Conf thấp hơn
        )

        snapped = self.snapper.snap_detections([det_1, det_2])
        self.assertEqual(len(snapped), 1)
        self.assertEqual(snapped[0].piece_info.class_id, NAME_TO_CLASS_ID["b_chariot"])

    def test_fen_generation_and_roundtrip(self):
        """Kiểm tra xuất FEN khởi đầu chuẩn và nạp lại từ FEN"""
        initial_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

        board = BoardState()
        board.load_from_fen(initial_fen)

        exported_fen = board.to_fen()
        self.assertEqual(exported_fen, initial_fen)

        # Kiểm tra quân cụ thể
        black_king = board.get_piece(col=4, row=0)
        self.assertIsNotNone(black_king)
        self.assertEqual(black_king.piece_info.fen_char, "k")

        red_king = board.get_piece(col=4, row=9)
        self.assertIsNotNone(red_king)
        self.assertEqual(red_king.piece_info.fen_char, "K")

    def test_piece_count_constraint_enforcement(self):
        """Kiểm tra lọc bỏ quân thừa vượt quá giới hạn luật (ví dụ: phát hiện 2 Tướng Đỏ)"""
        p_info_king = PIECE_REGISTRY[NAME_TO_CLASS_ID["r_king"]]

        from src.post_processing.grid_snapper import SnappedPiece
        king_1 = SnappedPiece(
            col=4, row=9, piece_info=p_info_king, confidence=0.99, det_confidence=1.0,
            distance_to_grid=1.0, center_x=0.0, center_y=0.0, bbox=(0, 0, 0, 0)
        )
        king_2 = SnappedPiece(
            col=4, row=8, piece_info=p_info_king, confidence=0.55, det_confidence=1.0,
            distance_to_grid=2.0, center_x=0.0, center_y=0.0, bbox=(0, 0, 0, 0)
        )

        board = BoardState(enforce_piece_limits=True)
        warnings = board.update_from_snapped_pieces([king_1, king_2])

        all_pieces = board.get_all_pieces()
        self.assertEqual(len(all_pieces), 1)
        self.assertEqual(all_pieces[0].col, 4)
        self.assertEqual(all_pieces[0].row, 9)
        self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
