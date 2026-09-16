"""
Package Dữ liệu tổng hợp (Synthetic Xiangqi Board & Piece Generation).
"""

from .board_renderer import (
    SyntheticPieceInstance,
    SyntheticBoardOutput,
    SyntheticXiangqiRenderer,
)
from .dataset_generator import SyntheticDatasetExporter

__all__ = [
    "SyntheticPieceInstance",
    "SyntheticBoardOutput",
    "SyntheticXiangqiRenderer",
    "SyntheticDatasetExporter",
]
