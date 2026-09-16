"""
Module Snap tọa độ quân cờ vào lưới giao điểm 9x10 (Grid Snapping).
Khớp tọa độ pixel của quân cờ từ Giai đoạn B vào các giao điểm lý thuyết đã biết trước.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from scipy.spatial import cKDTree

from .piece_definitions import (
    BOARD_NUM_COLS,
    BOARD_NUM_ROWS,
    PIECE_REGISTRY,
    PieceInfo,
)


@dataclass
class RawPieceDetection:
    """Thông tin 1 quân cờ phát hiện được trước khi snap"""
    center_x: float
    center_y: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 trên canvas chuẩn hóa
    class_id: int
    class_name: str
    confidence: float
    det_confidence: float = 1.0


@dataclass
class SnappedPiece:
    """Thông tin quân cờ sau khi đã snap vào giao điểm lưới (col, row)"""
    col: int                          # 0 đến 8 (cột)
    row: int                          # 0 đến 9 (hàng)
    piece_info: PieceInfo
    confidence: float
    det_confidence: float
    distance_to_grid: float           # Khoảng cách từ tâm phát hiện đến giao điểm lý thuyết (pixel)
    center_x: float
    center_y: float
    bbox: Tuple[float, float, float, float]


class GridSnapper:
    """
    Snap các tọa độ quân cờ phát hiện được vào 90 giao điểm chuẩn của bàn cờ tướng.
    """

    def __init__(
        self,
        canonical_grid_points: np.ndarray,
        max_snap_radius_px: float = 38.0,
        conflict_resolution: str = "highest_conf",
    ):
        """
        Args:
            canonical_grid_points: Mảng shape (10, 9, 2) hoặc (90, 2) chứa tọa độ (x, y) của 90 giao điểm.
            max_snap_radius_px: Bán kính pixel tối đa cho phép snap vào 1 giao điểm.
            conflict_resolution: Cách giải quyết xung đột khi có 2 quân snap vào cùng 1 điểm ('highest_conf' hoặc 'nearest').
        """
        self.max_snap_radius_px = float(max_snap_radius_px)
        self.conflict_resolution = conflict_resolution

        # Đảm bảo grid_points ở định dạng (10, 9, 2) và (90, 2)
        if canonical_grid_points.ndim == 3 and canonical_grid_points.shape[:2] == (BOARD_NUM_ROWS, BOARD_NUM_COLS):
            self.grid_2d = canonical_grid_points  # shape (10, 9, 2)
            self.grid_flat = canonical_grid_points.reshape(-1, 2)  # shape (90, 2)
        elif canonical_grid_points.ndim == 2 and canonical_grid_points.shape[0] == 90:
            self.grid_flat = canonical_grid_points
            self.grid_2d = canonical_grid_points.reshape(BOARD_NUM_ROWS, BOARD_NUM_COLS, 2)
        else:
            raise ValueError(f"Tọa độ lưới không hợp lệ: shape {canonical_grid_points.shape}, kỳ vọng (10, 9, 2) hoặc (90, 2)")

        # Tạo KD-Tree để truy vấn điểm gần nhất trong O(log N)
        self.kdtree = cKDTree(self.grid_flat)

    def snap_detections(
        self,
        raw_detections: List[RawPieceDetection]
    ) -> List[SnappedPiece]:
        """
        Khớp danh sách các quân cờ phát hiện được vào các giao điểm trên bàn cờ.
        Tự động lọc các quân nằm ngoài bán kính cho phép và xử lý xung đột (2 quân cùng 1 ô).
        """
        if not raw_detections:
            return []

        centers = np.array([[det.center_x, det.center_y] for det in raw_detections], dtype=np.float32)

        # Truy vấn khoảng cách và chỉ số giao điểm gần nhất
        distances, indices = self.kdtree.query(centers)

        # Gom nhóm các quân cờ theo giao điểm để giải quyết xung đột
        grid_occupied_candidates: Dict[int, List[Tuple[float, RawPieceDetection]]] = {}

        for i, (dist, idx) in enumerate(zip(distances, indices)):
            # Bỏ qua nếu lệch quá xa so với giao điểm lưới (nhiễu hoặc quân ngoài bàn)
            if dist > self.max_snap_radius_px:
                continue

            det = raw_detections[i]
            if idx not in grid_occupied_candidates:
                grid_occupied_candidates[idx] = []
            grid_occupied_candidates[idx].append((dist, det))

        # Giải quyết xung đột nếu nhiều quân snap vào cùng 1 giao điểm
        snapped_pieces: List[SnappedPiece] = []

        for idx, candidates in grid_occupied_candidates.items():
            row = int(idx // BOARD_NUM_COLS)
            col = int(idx % BOARD_NUM_COLS)

            if len(candidates) == 1:
                dist, best_det = candidates[0]
            else:
                if self.conflict_resolution == "nearest":
                    # Chọn ứng viên có khoảng cách đến tâm giao điểm nhỏ nhất
                    candidates.sort(key=lambda x: x[0])
                    dist, best_det = candidates[0]
                else:
                    # Mặc định: Chọn ứng viên có confidence phân loại cao nhất
                    candidates.sort(key=lambda x: (x[1].confidence * x[1].det_confidence), reverse=True)
                    dist, best_det = candidates[0]

            piece_info = PIECE_REGISTRY.get(best_det.class_id)
            if piece_info is None:
                continue

            snapped_pieces.append(
                SnappedPiece(
                    col=col,
                    row=row,
                    piece_info=piece_info,
                    confidence=best_det.confidence,
                    det_confidence=best_det.det_confidence,
                    distance_to_grid=float(dist),
                    center_x=best_det.center_x,
                    center_y=best_det.center_y,
                    bbox=best_det.bbox,
                )
            )

        return snapped_pieces
