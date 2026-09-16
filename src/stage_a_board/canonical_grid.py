"""
Định nghĩa hệ tọa độ lưới chuẩn hóa (Canonical Grid) cho bàn cờ tướng 9 cột x 10 hàng.
"""

from typing import List, Tuple
import numpy as np
import cv2

from ..post_processing.piece_definitions import BOARD_NUM_COLS, BOARD_NUM_ROWS


class CanonicalBoardGrid:
    """
    Quản lý tọa độ 90 giao điểm chuẩn trên ảnh canvas đã warp Homography.
    """

    def __init__(
        self,
        canvas_width: int = 720,
        canvas_height: int = 800,
        margin_x: int = 40,
        margin_y: int = 40,
    ):
        """
        Args:
            canvas_width: Chiều rộng canvas chuẩn hóa (pixel)
            canvas_height: Chiều cao canvas chuẩn hóa (pixel)
            margin_x: Khoảng lề trái/phải tính từ mép ảnh đến đường biên ngoài của bàn cờ
            margin_y: Khoảng lề trên/dưới tính từ mép ảnh đến đường biên ngoài của bàn cờ
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.margin_x = margin_x
        self.margin_y = margin_y

        self.num_cols = BOARD_NUM_COLS  # 9 cột (8 ô khoảng cách)
        self.num_rows = BOARD_NUM_ROWS  # 10 hàng (9 ô khoảng cách)

        # Tính toán bước nhảy lưới (grid step)
        self.grid_width = self.canvas_width - 2 * self.margin_x
        self.grid_height = self.canvas_height - 2 * self.margin_y

        self.step_x = self.grid_width / (self.num_cols - 1)
        self.step_y = self.grid_height / (self.num_rows - 1)

        # Tạo mảng tọa độ 90 giao điểm shape (10, 9, 2)
        self.grid_points_2d = self._build_grid_points()
        self.grid_points_flat = self.grid_points_2d.reshape(-1, 2)  # shape (90, 2)

        # 4 góc ngoài cùng của khung lưới bàn cờ (TL, TR, BR, BL)
        self.canonical_corners = np.array([
            [self.margin_x, self.margin_y],                                              # Top-Left (0,0)
            [self.canvas_width - self.margin_x, self.margin_y],                          # Top-Right (8,0)
            [self.canvas_width - self.margin_x, self.canvas_height - self.margin_y],     # Bottom-Right (8,9)
            [self.margin_x, self.canvas_height - self.margin_y],                         # Bottom-Left (0,9)
        ], dtype=np.float32)

    def _build_grid_points(self) -> np.ndarray:
        """Tạo lưới tọa độ (x, y) cho từng ô (row, col)"""
        grid = np.zeros((self.num_rows, self.num_cols, 2), dtype=np.float32)
        for r in range(self.num_rows):
            y = self.margin_y + r * self.step_y
            for c in range(self.num_cols):
                x = self.margin_x + c * self.step_x
                grid[r, c] = [x, y]
        return grid

    def get_pixel_coord(self, col: int, row: int) -> Tuple[float, float]:
        """Lấy tọa độ pixel (x, y) trên canvas của giao điểm (col, row)"""
        if not (0 <= col < self.num_cols and 0 <= row < self.num_rows):
            raise IndexError(f"Tọa độ ({col}, {row}) nằm ngoài bàn cờ 9x10")
        pt = self.grid_points_2d[row, col]
        return float(pt[0]), float(pt[1])

    def draw_grid_overlay(
        self,
        image: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1,
        draw_palace: bool = True,
        draw_river: bool = True,
    ) -> np.ndarray:
        """Vẽ lưới bàn cờ, sông và cửu cung lên ảnh chuẩn hóa để debug / visualize"""
        vis = image.copy()

        # 1. Vẽ các đường kẻ ngang
        for r in range(self.num_rows):
            pt1 = (int(self.grid_points_2d[r, 0, 0]), int(self.grid_points_2d[r, 0, 1]))
            pt2 = (int(self.grid_points_2d[r, -1, 0]), int(self.grid_points_2d[r, -1, 1]))
            cv2.line(vis, pt1, pt2, color, thickness)

        # 2. Vẽ các đường kẻ dọc
        # Lưu ý: Cờ Tướng không có đường dọc qua sông ở 7 cột giữa (từ hàng 4 đến hàng 5)
        for c in range(self.num_cols):
            if c == 0 or c == self.num_cols - 1:
                # 2 cột biên ngoài nối liền
                pt1 = (int(self.grid_points_2d[0, c, 0]), int(self.grid_points_2d[0, c, 1]))
                pt2 = (int(self.grid_points_2d[-1, c, 0]), int(self.grid_points_2d[-1, c, 1]))
                cv2.line(vis, pt1, pt2, color, thickness)
            else:
                # Nửa trên (hàng 0 -> 4)
                pt1 = (int(self.grid_points_2d[0, c, 0]), int(self.grid_points_2d[0, c, 1]))
                pt2 = (int(self.grid_points_2d[4, c, 0]), int(self.grid_points_2d[4, c, 1]))
                cv2.line(vis, pt1, pt2, color, thickness)
                # Nửa dưới (hàng 5 -> 9)
                pt3 = (int(self.grid_points_2d[5, c, 0]), int(self.grid_points_2d[5, c, 1]))
                pt4 = (int(self.grid_points_2d[9, c, 0]), int(self.grid_points_2d[9, c, 1]))
                cv2.line(vis, pt3, pt4, color, thickness)

        # 3. Vẽ 2 Cửu Cung (Palace) - Đường chéo
        if draw_palace:
            # Cửu cung trên (hàng 0..2, cột 3..5)
            cv2.line(vis, (int(self.grid_points_2d[0, 3, 0]), int(self.grid_points_2d[0, 3, 1])),
                          (int(self.grid_points_2d[2, 5, 0]), int(self.grid_points_2d[2, 5, 1])), color, thickness)
            cv2.line(vis, (int(self.grid_points_2d[0, 5, 0]), int(self.grid_points_2d[0, 5, 1])),
                          (int(self.grid_points_2d[2, 3, 0]), int(self.grid_points_2d[2, 3, 1])), color, thickness)

            # Cửu cung dưới (hàng 7..9, cột 3..5)
            cv2.line(vis, (int(self.grid_points_2d[7, 3, 0]), int(self.grid_points_2d[7, 3, 1])),
                          (int(self.grid_points_2d[9, 5, 0]), int(self.grid_points_2d[9, 5, 1])), color, thickness)
            cv2.line(vis, (int(self.grid_points_2d[7, 5, 0]), int(self.grid_points_2d[7, 5, 1])),
                          (int(self.grid_points_2d[9, 3, 0]), int(self.grid_points_2d[9, 3, 1])), color, thickness)

        # 4. Vẽ các chấm giao điểm
        for pt in self.grid_points_flat:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

        return vis
