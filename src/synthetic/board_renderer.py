"""
Module sinh dữ liệu tổng hợp (Synthetic Xiangqi Board & Piece Renderer).
Tạo ảnh bàn cờ và quân cờ chân thực với đầy đủ nhãn ground-truth cho Stage A, B, C.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

from ..post_processing.piece_definitions import (
    BOARD_NUM_COLS,
    BOARD_NUM_ROWS,
    PIECE_REGISTRY,
    PieceColor,
    PieceInfo,
)
from ..stage_a_board.canonical_grid import CanonicalBoardGrid


@dataclass
class SyntheticPieceInstance:
    """Thông tin 1 quân cờ được sinh ra"""
    col: int
    row: int
    piece_info: PieceInfo
    canonical_center: Tuple[float, float]
    canonical_bbox: Tuple[float, float, float, float]
    projected_center: Optional[Tuple[float, float]] = None
    projected_bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class SyntheticBoardOutput:
    """Đầu ra hoàn chỉnh của 1 mẫu dữ liệu tổng hợp"""
    canonical_image: np.ndarray             # Ảnh chuẩn hóa (Top-Down)
    projected_image: np.ndarray             # Ảnh chụp phối cảnh từ góc camera nghiêng
    corners_canonical: np.ndarray           # 4 góc trên ảnh chuẩn hóa (4, 2)
    corners_projected: np.ndarray           # 4 góc trên ảnh phối cảnh (4, 2) [TL, TR, BR, BL]
    corners_visibility: np.ndarray          # Trạng thái nhìn thấy (4,) (1.0 hoặc 0.0 nếu bị che)
    pieces: List[SyntheticPieceInstance]    # Danh sách quân cờ cùng tọa độ & nhãn
    homography_matrix: np.ndarray           # Ma trận biến đổi phối cảnh


class SyntheticXiangqiRenderer:
    """
    Sinh ảnh bàn cờ tướng 2D/3D phục vụ huấn luyện Pre-train và Benchmark hệ thống.
    """

    def __init__(
        self,
        grid: Optional[CanonicalBoardGrid] = None,
        font_path: Optional[str] = None,
    ):
        self.grid = grid or CanonicalBoardGrid(canvas_width=720, canvas_height=800, margin_x=40, margin_y=40)
        self.font_path = font_path

    def _get_pil_font(self, size: int) -> ImageFont.ImageFont:
        """Tải font chữ Hán hoặc font mặc định hệ thống"""
        # Thử một số font tiếng Hoa / CJK phổ biến trên Windows
        windows_cjk_fonts = [
            "simsun.ttc", "simhei.ttf", "msyh.ttc", "kaiu.ttf", "mingliu.ttc",
            "C:\\Windows\\Fonts\\simsun.ttc", "C:\\Windows\\Fonts\\simhei.ttf",
            "C:\\Windows\\Fonts\\msyh.ttc", "C:\\Windows\\Fonts\\arial.ttf"
        ]

        if self.font_path and os.path.exists(self.font_path):
            try:
                return ImageFont.truetype(self.font_path, size)
            except Exception:
                pass

        for f in windows_cjk_fonts:
            try:
                return ImageFont.truetype(f, size)
            except Exception:
                continue

        return ImageFont.load_default()

    def _generate_board_canvas(self) -> Image.Image:
        """Tạo ảnh nền bàn cờ với màu gỗ/giấy vàng và đường kẻ"""
        w, h = self.grid.canvas_width, self.grid.canvas_height
        # Màu nền gỗ vàng ấm
        base_color = (
            random.randint(220, 240),
            random.randint(185, 205),
            random.randint(130, 155),
        )
        img = Image.new("RGB", (w, h), color=base_color)
        draw = ImageDraw.Draw(img)

        # Thêm vân gỗ nhẹ bằng nhiễu ngẫu nhiên
        noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
        img_arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)

        line_color = (50, 30, 15)
        line_w = 2

        # Vẽ đường ngang
        for r in range(self.grid.num_rows):
            x1, y1 = self.grid.get_pixel_coord(0, r)
            x2, y2 = self.grid.get_pixel_coord(self.grid.num_cols - 1, r)
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_w)

        # Vẽ đường dọc (ngắt ở sông)
        for c in range(self.grid.num_cols):
            if c == 0 or c == self.grid.num_cols - 1:
                x1, y1 = self.grid.get_pixel_coord(c, 0)
                x2, y2 = self.grid.get_pixel_coord(c, self.grid.num_rows - 1)
                draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_w)
            else:
                # Nửa trên
                x1, y1 = self.grid.get_pixel_coord(c, 0)
                x2, y2 = self.grid.get_pixel_coord(c, 4)
                draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_w)
                # Nửa dưới
                x3, y3 = self.grid.get_pixel_coord(c, 5)
                x4, y4 = self.grid.get_pixel_coord(c, 9)
                draw.line([(x3, y3), (x4, y4)], fill=line_color, width=line_w)

        # Cửu cung
        c0_3, r0_3 = self.grid.get_pixel_coord(3, 0)
        c2_5, r2_5 = self.grid.get_pixel_coord(5, 2)
        c0_5, r0_5 = self.grid.get_pixel_coord(5, 0)
        c2_3, r2_3 = self.grid.get_pixel_coord(3, 2)
        draw.line([(c0_3, r0_3), (c2_5, r2_5)], fill=line_color, width=line_w)
        draw.line([(c0_5, r0_5), (c2_3, r2_3)], fill=line_color, width=line_w)

        c7_3, r7_3 = self.grid.get_pixel_coord(3, 7)
        c9_5, r9_5 = self.grid.get_pixel_coord(5, 9)
        c7_5, r7_5 = self.grid.get_pixel_coord(5, 7)
        c9_3, r9_3 = self.grid.get_pixel_coord(3, 9)
        draw.line([(c7_3, r7_3), (c9_5, r9_5)], fill=line_color, width=line_w)
        draw.line([(c7_5, r7_5), (c9_3, r9_3)], fill=line_color, width=line_w)

        return img

    def _render_piece_patch(
        self,
        piece_info: PieceInfo,
        radius: int = 30,
        rotation_deg: float = 0.0,
    ) -> Image.Image:
        """Vẽ một quân cờ tròn với viền gỗ và ký tự chữ Hán"""
        diameter = radius * 2
        patch = Image.new("RGBA", (diameter + 4, diameter + 4), (0, 0, 0, 0))
        draw = ImageDraw.Draw(patch)

        cx, cy = diameter // 2 + 2, diameter // 2 + 2

        # 1. Bóng đổ (Shadow)
        draw.ellipse([cx - radius + 2, cy - radius + 2, cx + radius + 2, cy + radius + 2], fill=(40, 25, 10, 80))

        # 2. Thân quân cờ gỗ
        wood_body = (
            random.randint(235, 250),
            random.randint(210, 225),
            random.randint(170, 185),
            255
        )
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=wood_body, outline=(60, 35, 15, 255), width=2)

        # 3. Vòng tròn rãnh chìm phía trong
        inner_r = radius - 4
        draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r], outline=(120, 80, 40, 180), width=1)

        # 4. Ký tự chữ Hán
        font_size = int(radius * 1.15)
        font = self._get_pil_font(font_size)

        text_color = (190, 25, 25, 255) if piece_info.color == PieceColor.RED else (20, 20, 20, 255)
        char = piece_info.chinese_char

        # Canh giữa chữ
        bbox = draw.textbbox((0, 0), char, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = cx - tw / 2.0 - bbox[0]
        ty = cy - th / 2.0 - bbox[1]
        draw.text((tx, ty), char, font=font, fill=text_color)

        # Xoay ngẫu nhiên
        if rotation_deg != 0.0:
            patch = patch.rotate(rotation_deg, resample=Image.BICUBIC, expand=False)

        return patch

    def generate_random_board(
        self,
        board_state_fen: Optional[str] = None,
        piece_jitter_px: float = 3.0,
        occlude_one_corner: bool = False,
    ) -> SyntheticBoardOutput:
        """
        Sinh 1 bàn cờ ngẫu nhiên hoặc theo chuỗi FEN kèm phép chiếu 3D phối cảnh.
        """
        board_img = self._generate_board_canvas()
        grid = self.grid

        # 1. Khởi tạo danh sách quân cờ
        from ..post_processing.board_state import BoardState
        bs = BoardState()
        if board_state_fen:
            bs.load_from_fen(board_state_fen)
        else:
            # Mặc định tạo bàn cờ ban đầu chuẩn
            init_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
            bs.load_from_fen(init_fen)

        pieces_list: List[SyntheticPieceInstance] = []
        piece_radius = 28

        # 2. Vẽ từng quân cờ lên canvas chuẩn hóa
        for r in range(BOARD_NUM_ROWS):
            for c in range(BOARD_NUM_COLS):
                p = bs.grid[r][c]
                if p is not None:
                    base_x, base_y = grid.get_pixel_coord(c, r)
                    # Thêm độ lệch nhẹ (jitter) thực tế khi người đặt quân không chính xác tuyệt đối
                    jx = base_x + random.uniform(-piece_jitter_px, piece_jitter_px)
                    jy = base_y + random.uniform(-piece_jitter_px, piece_jitter_px)

                    # Xoay ngẫu nhiên
                    rot = random.uniform(-180.0, 180.0)
                    patch = self._render_piece_patch(p.piece_info, radius=piece_radius, rotation_deg=rot)

                    # Dán lên board
                    pw, ph = patch.size
                    px = int(round(jx - pw / 2.0))
                    py = int(round(jy - ph / 2.0))
                    board_img.paste(patch, (px, py), patch)

                    bbox_canonical = (
                        float(jx - piece_radius),
                        float(jy - piece_radius),
                        float(jx + piece_radius),
                        float(jy + piece_radius),
                    )

                    pieces_list.append(
                        SyntheticPieceInstance(
                            col=c,
                            row=r,
                            piece_info=p.piece_info,
                            canonical_center=(float(jx), float(jy)),
                            canonical_bbox=bbox_canonical,
                        )
                    )

        canonical_np = np.array(board_img)

        # 3. Tạo phép chiếu phối cảnh 3D ngẫu nhiên (Projective Perspective Transform)
        w, h = grid.canvas_width, grid.canvas_height
        src_corners = grid.canonical_corners.copy()  # [TL, TR, BR, BL]

        # Kích thước ảnh phối cảnh (Camera view)
        cam_w, cam_h = 1024, 768
        pad_x, pad_y = 120, 100

        # Tạo 4 góc nghiêng tự nhiên mô phỏng góc chụp của người dùng (từ trên nhìn xéo xuống)
        tilt_top = random.uniform(60, 120)
        dst_tl = [pad_x + tilt_top + random.uniform(-20, 20), pad_y + random.uniform(-20, 20)]
        dst_tr = [cam_w - pad_x - tilt_top + random.uniform(-20, 20), pad_y + random.uniform(-20, 20)]
        dst_br = [cam_w - pad_x + random.uniform(-20, 20), cam_h - pad_y + random.uniform(-20, 20)]
        dst_bl = [pad_x + random.uniform(-20, 20), cam_h - pad_y + random.uniform(-20, 20)]

        dst_corners = np.array([dst_tl, dst_tr, dst_br, dst_bl], dtype=np.float32)

        # Ma trận biến đổi từ Canonical -> Perspective Camera
        H_forward = cv2.getPerspectiveTransform(src_corners, dst_corners)

        # Tạo ảnh nền bàn phòng / vải bàn
        table_bg = np.full((cam_h, cam_w, 3), (random.randint(60, 90), random.randint(50, 75), random.randint(40, 60)), dtype=np.uint8)

        # Warp bàn cờ vào ảnh camera
        warped_board = cv2.warpPerspective(canonical_np, H_forward, (cam_w, cam_h))
        mask = cv2.warpPerspective(np.full((h, w), 255, dtype=np.uint8), H_forward, (cam_w, cam_h))

        projected_image = table_bg.copy()
        mask_3ch = cv2.merge([mask, mask, mask]) > 0
        projected_image[mask_3ch] = warped_board[mask_3ch]

        # Tính tọa độ các quân cờ trên ảnh phối cảnh
        for piece in pieces_list:
            pt = np.array([[piece.canonical_center]], dtype=np.float32)
            proj_pt = cv2.perspectiveTransform(pt, H_forward)[0, 0]
            piece.projected_center = (float(proj_pt[0]), float(proj_pt[1]))

            # Chiếu 4 góc bbox
            bx1, by1, bx2, by2 = piece.canonical_bbox
            box_pts = np.array([[[bx1, by1]], [[bx2, by1]], [[bx2, by2]], [[bx1, by2]]], dtype=np.float32)
            proj_box_pts = cv2.perspectiveTransform(box_pts, H_forward).reshape(-1, 2)
            min_x, min_y = np.min(proj_box_pts, axis=0)
            max_x, max_y = np.max(proj_box_pts, axis=0)
            piece.projected_bbox = (float(min_x), float(min_y), float(max_x), float(max_y))

        # Xử lý che góc ngẫu nhiên (Occlusion flag)
        visibilities = np.ones(4, dtype=np.float32)
        if occlude_one_corner:
            occ_idx = random.randint(0, 3)
            visibilities[occ_idx] = 0.0
            # Vẽ hình tròn che góc mô phỏng tay người hoặc tách trà
            occ_pt = dst_corners[occ_idx].astype(int)
            cv2.circle(projected_image, tuple(occ_pt), random.randint(40, 60), (30, 20, 15), -1)

        return SyntheticBoardOutput(
            canonical_image=canonical_np,
            projected_image=projected_image,
            corners_canonical=src_corners,
            corners_projected=dst_corners,
            corners_visibility=visibilities,
            pieces=pieces_list,
            homography_matrix=H_forward,
        )
