"""
Package Giai đoạn A: Định vị & Chuẩn hóa bàn cờ (Stage A - Board Localization & Rectification).
"""

from .canonical_grid import CanonicalBoardGrid
from .rectification import CameraCalibrator, HomographyRectifier
from .corner_detector import (
    BoardCornerDetector,
    BoardCornersResult,
    order_corners_clockwise,
    recover_missing_corner,
)

__all__ = [
    "CanonicalBoardGrid",
    "CameraCalibrator",
    "HomographyRectifier",
    "BoardCornerDetector",
    "BoardCornersResult",
    "order_corners_clockwise",
    "recover_missing_corner",
]
