"""
Định nghĩa các thông số, hằng số, lớp quân cờ tướng và quy tắc chuẩn.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PieceColor(str, Enum):
    RED = "red"
    BLACK = "black"


class PieceType(str, Enum):
    KING = "king"          # Tướng
    ADVISOR = "advisor"    # Sĩ
    ELEPHANT = "elephant"  # Tượng
    HORSE = "horse"        # Mã
    CHARIOT = "chariot"    # Xe
    CANNON = "cannon"      # Pháo
    SOLDIER = "soldier"    # Tốt / Binh


@dataclass(frozen=True)
class PieceInfo:
    class_id: int
    name_code: str
    color: PieceColor
    piece_type: PieceType
    chinese_char: str
    vietnamese_name: str
    fen_char: str          # Chữ hoa cho Đỏ, chữ thường cho Đen
    max_count: int         # Số lượng tối đa trong một ván cờ cho 1 bên


# Danh mục 14 lớp quân cờ (7 loại x 2 màu)
PIECE_REGISTRY: Dict[int, PieceInfo] = {
    0: PieceInfo(
        class_id=0,
        name_code="r_king",
        color=PieceColor.RED,
        piece_type=PieceType.KING,
        chinese_char="帥",
        vietnamese_name="Tướng Đỏ",
        fen_char="K",
        max_count=1,
    ),
    1: PieceInfo(
        class_id=1,
        name_code="r_advisor",
        color=PieceColor.RED,
        piece_type=PieceType.ADVISOR,
        chinese_char="仕",
        vietnamese_name="Sĩ Đỏ",
        fen_char="A",
        max_count=2,
    ),
    2: PieceInfo(
        class_id=2,
        name_code="r_elephant",
        color=PieceColor.RED,
        piece_type=PieceType.ELEPHANT,
        chinese_char="相",
        vietnamese_name="Tượng Đỏ",
        fen_char="B",  # Trong FEN Cờ Tướng thường dùng B (Bishop) hoặc E
        max_count=2,
    ),
    3: PieceInfo(
        class_id=3,
        name_code="r_horse",
        color=PieceColor.RED,
        piece_type=PieceType.HORSE,
        chinese_char="傌",
        vietnamese_name="Mã Đỏ",
        fen_char="N",  # Trong FEN thường dùng N (Knight) hoặc H
        max_count=2,
    ),
    4: PieceInfo(
        class_id=4,
        name_code="r_chariot",
        color=PieceColor.RED,
        piece_type=PieceType.CHARIOT,
        chinese_char="俥",
        vietnamese_name="Xe Đỏ",
        fen_char="R",  # R (Rook)
        max_count=2,
    ),
    5: PieceInfo(
        class_id=5,
        name_code="r_cannon",
        color=PieceColor.RED,
        piece_type=PieceType.CANNON,
        chinese_char="炮",
        vietnamese_name="Pháo Đỏ",
        fen_char="C",  # C (Cannon)
        max_count=2,
    ),
    6: PieceInfo(
        class_id=6,
        name_code="r_soldier",
        color=PieceColor.RED,
        piece_type=PieceType.SOLDIER,
        chinese_char="兵",
        vietnamese_name="Binh/Tốt Đỏ",
        fen_char="P",  # P (Pawn)
        max_count=5,
    ),
    7: PieceInfo(
        class_id=7,
        name_code="b_king",
        color=PieceColor.BLACK,
        piece_type=PieceType.KING,
        chinese_char="將",
        vietnamese_name="Tướng Đen",
        fen_char="k",
        max_count=1,
    ),
    8: PieceInfo(
        class_id=8,
        name_code="b_advisor",
        color=PieceColor.BLACK,
        piece_type=PieceType.ADVISOR,
        chinese_char="士",
        vietnamese_name="Sĩ Đen",
        fen_char="a",
        max_count=2,
    ),
    9: PieceInfo(
        class_id=9,
        name_code="b_elephant",
        color=PieceColor.BLACK,
        piece_type=PieceType.ELEPHANT,
        chinese_char="象",
        vietnamese_name="Tượng Đen",
        fen_char="b",
        max_count=2,
    ),
    10: PieceInfo(
        class_id=10,
        name_code="b_horse",
        color=PieceColor.BLACK,
        piece_type=PieceType.HORSE,
        chinese_char="馬",
        vietnamese_name="Mã Đen",
        fen_char="n",
        max_count=2,
    ),
    11: PieceInfo(
        class_id=11,
        name_code="b_chariot",
        color=PieceColor.BLACK,
        piece_type=PieceType.CHARIOT,
        chinese_char="車",
        vietnamese_name="Xe Đen",
        fen_char="r",
        max_count=2,
    ),
    12: PieceInfo(
        class_id=12,
        name_code="b_cannon",
        color=PieceColor.BLACK,
        piece_type=PieceType.CANNON,
        chinese_char="砲",
        vietnamese_name="Pháo Đen",
        fen_char="c",
        max_count=2,
    ),
    13: PieceInfo(
        class_id=13,
        name_code="b_soldier",
        color=PieceColor.BLACK,
        piece_type=PieceType.SOLDIER,
        chinese_char="卒",
        vietnamese_name="Tốt Đen",
        fen_char="p",
        max_count=5,
    ),
}

NAME_TO_CLASS_ID: Dict[str, int] = {info.name_code: info.class_id for info in PIECE_REGISTRY.values()}
FEN_TO_CLASS_ID: Dict[str, int] = {info.fen_char: info.class_id for info in PIECE_REGISTRY.values()}
CHAR_TO_CLASS_ID: Dict[str, int] = {info.chinese_char: info.class_id for info in PIECE_REGISTRY.values()}

MAX_PIECE_LIMITS: Dict[str, int] = {
    "king": 1,
    "advisor": 2,
    "elephant": 2,
    "horse": 2,
    "chariot": 2,
    "cannon": 2,
    "soldier": 5,
}

# 90 giao điểm chuẩn của bàn cờ
BOARD_NUM_COLS = 9
BOARD_NUM_ROWS = 10
BOARD_TOTAL_INTERSECTIONS = 90
