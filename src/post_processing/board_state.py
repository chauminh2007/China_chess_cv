"""
Quản lý trạng thái bàn cờ (Board State), áp dụng ràng buộc luật Cờ Tướng và xuất chuỗi FEN chuẩn.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .piece_definitions import (
    BOARD_NUM_COLS,
    BOARD_NUM_ROWS,
    PIECE_REGISTRY,
    PieceColor,
    PieceInfo,
    PieceType,
    MAX_PIECE_LIMITS,
)
from .grid_snapper import SnappedPiece


class BoardState:
    """
    Biểu diễn trạng thái toàn cục của bàn cờ 10 hàng x 9 cột (90 giao điểm).
    Tự động áp dụng ràng buộc luật cờ và xuất FEN.
    """

    def __init__(self, enforce_piece_limits: bool = True):
        self.enforce_piece_limits = enforce_piece_limits
        # Ma trận 10 hàng x 9 cột, mỗi ô là SnappedPiece hoặc None
        self.grid: List[List[Optional[SnappedPiece]]] = [
            [None for _ in range(BOARD_NUM_COLS)] for _ in range(BOARD_NUM_ROWS)
        ]
        self.active_turn: PieceColor = PieceColor.RED

    def clear(self):
        """Xóa trắng bàn cờ"""
        self.grid = [[None for _ in range(BOARD_NUM_COLS)] for _ in range(BOARD_NUM_ROWS)]

    def update_from_snapped_pieces(
        self,
        pieces: List[SnappedPiece],
        enforce_limits: Optional[bool] = None,
    ) -> List[str]:
        """
        Cập nhật ma trận bàn cờ từ danh sách quân đã snap.
        Áp dụng lọc ràng buộc số lượng tối đa và trả về danh sách cảnh báo/sửa lỗi (nếu có).
        """
        self.clear()
        do_enforce = self.enforce_piece_limits if enforce_limits is None else enforce_limits
        warnings: List[str] = []

        valid_pieces = pieces.copy()

        # Áp dụng ràng buộc số lượng quân tối đa theo luật
        if do_enforce:
            # Gom nhóm theo (color, piece_type)
            grouped_pieces: Dict[Tuple[PieceColor, PieceType], List[SnappedPiece]] = {}
            for p in valid_pieces:
                key = (p.piece_info.color, p.piece_info.piece_type)
                if key not in grouped_pieces:
                    grouped_pieces[key] = []
                grouped_pieces[key].append(p)

            filtered_pieces: List[SnappedPiece] = []
            for (color, p_type), group in grouped_pieces.items():
                limit = MAX_PIECE_LIMITS.get(p_type.value, 2)
                if len(group) > limit:
                    # Sắp xếp giảm dần theo confidence phân loại
                    group.sort(key=lambda x: (x.confidence * x.det_confidence), reverse=True)
                    kept = group[:limit]
                    rejected = group[limit:]
                    filtered_pieces.extend(kept)

                    for r in rejected:
                        warnings.append(
                            f"Lọc bỏ quân thừa do vượt giới hạn luật: {r.piece_info.vietnamese_name} "
                            f"tại ({r.col}, {r.row}) với conf={r.confidence:.2f}"
                        )
                else:
                    filtered_pieces.extend(group)

            valid_pieces = filtered_pieces

        # Đặt quân vào ma trận
        for p in valid_pieces:
            if 0 <= p.row < BOARD_NUM_ROWS and 0 <= p.col < BOARD_NUM_COLS:
                self.grid[p.row][p.col] = p

        return warnings

    def get_piece(self, col: int, row: int) -> Optional[SnappedPiece]:
        """Lấy thông tin quân cờ tại tọa độ (col, row)"""
        if 0 <= row < BOARD_NUM_ROWS and 0 <= col < BOARD_NUM_COLS:
            return self.grid[row][col]
        return None

    def get_all_pieces(self) -> List[SnappedPiece]:
        """Lấy danh sách tất cả các quân cờ hiện có trên bàn"""
        pieces = []
        for r in range(BOARD_NUM_ROWS):
            for c in range(BOARD_NUM_COLS):
                p = self.grid[r][c]
                if p is not None:
                    pieces.append(p)
        return pieces

    def to_matrix_char(self) -> List[List[str]]:
        """Xuất ma trận ký tự 10x9 (Chữ Hán hoặc '.')"""
        mat = []
        for r in range(BOARD_NUM_ROWS):
            row_chars = []
            for c in range(BOARD_NUM_COLS):
                p = self.grid[r][c]
                if p is None:
                    row_chars.append(" . ")
                else:
                    row_chars.append(f"{p.piece_info.chinese_char} ")
            mat.append(row_chars)
        return mat

    def to_fen(self, active_color: str = "w") -> str:
        """
        Chuyển đổi trạng thái bàn cờ sang chuỗi FEN chuẩn Cờ Tướng (Xiangqi FEN).
        Quy ước FEN Cờ Tướng:
        - Hàng 0 đến Hàng 9 (từ trên xuống dưới)
        - Chữ hoa: Đỏ (W/Red), Chữ thường: Đen (B/Black)
        - Ô trống liên tiếp được gom thành số
        Ví dụ khởi đầu: rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
        """
        fen_rows = []
        for r in range(BOARD_NUM_ROWS):
            empty_count = 0
            row_str = ""
            for c in range(BOARD_NUM_COLS):
                p = self.grid[r][c]
                if p is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += p.piece_info.fen_char
            if empty_count > 0:
                row_str += str(empty_count)
            fen_rows.append(row_str)

        board_fen = "/".join(fen_rows)
        # Bổ sung các thông số FEN: [active color] [castling] [en passant] [halfmove] [fullmove]
        full_fen = f"{board_fen} {active_color} - - 0 1"
        return full_fen

    def load_from_fen(self, fen: str):
        """Khôi phục trạng thái bàn cờ từ chuỗi FEN"""
        self.clear()
        board_part = fen.split()[0]
        rows = board_part.split("/")
        if len(rows) != BOARD_NUM_ROWS:
            raise ValueError(f"FEN không hợp lệ: số hàng là {len(rows)}, kỳ vọng 10")

        from .piece_definitions import FEN_TO_CLASS_ID

        for r_idx, row_str in enumerate(rows):
            col_idx = 0
            for char in row_str:
                if char.isdigit():
                    col_idx += int(char)
                else:
                    class_id = FEN_TO_CLASS_ID.get(char)
                    if class_id is not None:
                        piece_info = PIECE_REGISTRY[class_id]
                        snapped = SnappedPiece(
                            col=col_idx,
                            row=r_idx,
                            piece_info=piece_info,
                            confidence=1.0,
                            det_confidence=1.0,
                            distance_to_grid=0.0,
                            center_x=0.0,
                            center_y=0.0,
                            bbox=(0.0, 0.0, 0.0, 0.0),
                        )
                        self.grid[r_idx][col_idx] = snapped
                    col_idx += 1

    def to_ascii_display(self) -> str:
        """Tạo chuỗi hiển thị bàn cờ đẹp mắt trên console/terminal"""
        lines = []
        lines.append("  " + "   ".join([str(c) for c in range(BOARD_NUM_COLS)]))
        lines.append("  " + "---" * BOARD_NUM_COLS)
        for r in range(BOARD_NUM_ROWS):
            row_items = []
            for c in range(BOARD_NUM_COLS):
                p = self.grid[r][c]
                if p is None:
                    row_items.append(" + ")
                else:
                    row_items.append(f"{p.piece_info.chinese_char} ")
            lines.append(f"{r}| " + " ".join(row_items))
            if r == 4:
                lines.append("   " + "=== 楚 河 === 漢 界 ===")
        return "\n".join(lines)
