"""
Bộ xuất dữ liệu tổng hợp đồng bộ cho cả 3 giai đoạn (Stage A, B, C).
Tự động chia train/val theo session để tránh data leakage.
"""

from typing import Optional, List, Dict
import os
import random
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

from .board_renderer import SyntheticXiangqiRenderer, SyntheticBoardOutput
from ..post_processing.piece_definitions import NAME_TO_CLASS_ID, PIECE_REGISTRY


class SyntheticDatasetExporter:
    """
    Quản lý sinh hàng loạt và xuất dataset cho từng giai đoạn A, B, C.
    """

    def __init__(
        self,
        output_root: str = "data/synthetic",
        renderer: Optional[SyntheticXiangqiRenderer] = None,
    ):
        self.output_root = output_root
        self.renderer = renderer or SyntheticXiangqiRenderer()

        # Tạo cấu trúc thư mục
        self.stage_a_dir = os.path.join(output_root, "stage_a_board")
        self.stage_b_dir = os.path.join(output_root, "stage_b_pieces")
        self.stage_c_dir = os.path.join(output_root, "stage_c_classifier")

    def _setup_yolo_dirs(self, base_dir: str):
        """Tạo các thư mục images và labels cho YOLO"""
        for split in ("train", "val"):
            os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)

    def _setup_stage_c_dirs(self):
        """Tạo cấu trúc thư mục phân loại ImageFolder cho Stage C"""
        for split in ("train", "val"):
            for class_name in NAME_TO_CLASS_ID.keys():
                os.makedirs(os.path.join(self.stage_c_dir, split, class_name), exist_ok=True)

    def generate_full_dataset(
        self,
        num_train_boards: int = 100,
        num_val_boards: int = 20,
        occlusion_prob: float = 0.25,
    ):
        """
        Sinh trọn bộ dataset đồng bộ cho Stage A, B, C.
        """
        print(f"Bắt đầu sinh dữ liệu tổng hợp tại {self.output_root}...")
        self._setup_yolo_dirs(self.stage_a_dir)
        self._setup_yolo_dirs(self.stage_b_dir)
        self._setup_stage_c_dirs()

        for split, count in [("train", num_train_boards), ("val", num_val_boards)]:
            print(f"-> Đang sinh {count} bàn cờ cho tập {split}...")
            for i in tqdm(range(count), desc=f"Generating {split}"):
                sample_id = f"{split}_{i:05d}"
                occlude = (random.random() < occlusion_prob)

                # Sinh mẫu bàn cờ
                out: SyntheticBoardOutput = self.renderer.generate_random_board(
                    occlude_one_corner=occlude
                )

                # 1. Lưu dữ liệu Stage A (YOLO Pose)
                self._save_stage_a_sample(out, sample_id, split)

                # 2. Lưu dữ liệu Stage B (YOLO Detection trên ảnh chuẩn hóa)
                self._save_stage_b_sample(out, sample_id, split)

                # 3. Lưu dữ liệu Stage C (Patch classification)
                self._save_stage_c_patches(out, sample_id, split)

        print(f"Hoàn thành sinh dataset! Dữ liệu đã được lưu tại {self.output_root}")

    def _save_stage_a_sample(self, out: SyntheticBoardOutput, sample_id: str, split: str):
        """Lưu ảnh phối cảnh và nhãn 4 góc (YOLO-Pose format)"""
        img_path = os.path.join(self.stage_a_dir, "images", split, f"{sample_id}.jpg")
        lbl_path = os.path.join(self.stage_a_dir, "labels", split, f"{sample_id}.txt")

        # Lưu ảnh BGR
        cv2.imwrite(img_path, out.projected_image[..., ::-1])

        h, w = out.projected_image.shape[:2]
        pts = out.corners_projected  # shape (4, 2)
        vis = out.corners_visibility  # shape (4,)

        # Bounding box bao toàn bộ 4 góc bàn
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        box_cx = ((min_x + max_x) / 2.0) / w
        box_cy = ((min_y + max_y) / 2.0) / h
        box_w = (max_x - min_x) / w
        box_h = (max_y - min_y) / h

        # Keypoints: [x/w, y/h, v]
        kps_str = []
        for (kx, ky), v in zip(pts, vis):
            v_flag = 2 if v > 0.5 else 1  # 2: visible, 1: occluded
            kps_str.append(f"{kx / w:.6f} {ky / h:.6f} {v_flag}")

        line = f"0 {box_cx:.6f} {box_cy:.6f} {box_w:.6f} {box_h:.6f} " + " ".join(kps_str)
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(line + "\n")

    def _save_stage_b_sample(self, out: SyntheticBoardOutput, sample_id: str, split: str):
        """Lưu ảnh chuẩn hóa và nhãn bounding box của từng quân cờ (YOLO Detection format)"""
        img_path = os.path.join(self.stage_b_dir, "images", split, f"{sample_id}.jpg")
        lbl_path = os.path.join(self.stage_b_dir, "labels", split, f"{sample_id}.txt")

        cv2.imwrite(img_path, out.canonical_image[..., ::-1])

        h, w = out.canonical_image.shape[:2]
        lines = []
        for piece in out.pieces:
            x1, y1, x2, y2 = piece.canonical_bbox
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # Lớp 0: piece
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _save_stage_c_patches(self, out: SyntheticBoardOutput, sample_id: str, split: str):
        """Cắt và lưu patch từng quân cờ vào thư mục phân loại 14 lớp"""
        canonical_img = out.canonical_image
        h, w = canonical_img.shape[:2]

        for p_idx, piece in enumerate(out.pieces):
            x1, y1, x2, y2 = piece.canonical_bbox
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            r = int(round((x2 - x1) / 2.0)) + 3

            px1 = max(0, cx - r)
            py1 = max(0, cy - r)
            px2 = min(w, cx + r)
            py2 = min(h, cy + r)

            patch = canonical_img[py1:py2, px1:px2]
            if patch.size == 0:
                continue

            patch_resized = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)

            class_name = piece.piece_info.name_code
            save_name = f"{sample_id}_p{p_idx:02d}.png"
            save_path = os.path.join(self.stage_c_dir, split, class_name, save_name)

            cv2.imwrite(save_path, patch_resized[..., ::-1])
