"""
Package Hậu xử lý: Snap lưới, Ràng buộc luật và FEN.
"""

from .piece_definitions import (
    PieceColor,
    PieceType,
    PieceInfo,
    PIECE_REGISTRY,
    NAME_TO_CLASS_ID,
    FEN_TO_CLASS_ID,
    CHAR_TO_CLASS_ID,
    MAX_PIECE_LIMITS,
    BOARD_NUM_COLS,
    BOARD_NUM_ROWS,
    BOARD_TOTAL_INTERSECTIONS,
)
from .grid_snapper import (
    RawPieceDetection,
    SnappedPiece,
    GridSnapper,
)
from .board_state import (
    BoardState,
)

__all__ = [
    "PieceColor",
    "PieceType",
    "PieceInfo",
    "PIECE_REGISTRY",
    "NAME_TO_CLASS_ID",
    "FEN_TO_CLASS_ID",
    "CHAR_TO_CLASS_ID",
    "MAX_PIECE_LIMITS",
    "BOARD_NUM_COLS",
    "BOARD_NUM_ROWS",
    "BOARD_TOTAL_INTERSECTIONS",
    "RawPieceDetection",
    "SnappedPiece",
    "GridSnapper",
    "BoardState",
]
