"""
Phát hiện 4 góc bàn cờ (YOLO-Pose Corner Detector) và Thuật toán khôi phục góc bị che.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import cv2


@dataclass
class BoardCornersResult:
    """Kết quả phát hiện 4 góc bàn cờ"""
    corners: np.ndarray             # shape (4, 2) theo thứ tự [TL, TR, BR, BL]
    confidences: np.ndarray         # shape (4,) confidence của từng góc
    is_recovered: bool              # True nếu có 1 góc được phục hồi từ hình học
    recovered_corner_index: Optional[int] = None  # Chỉ số góc (0..3) được phục hồi


def order_corners_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 điểm bất kỳ theo thứ tự chuẩn: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    """
    pts = np.array(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Kỳ vọng mảng (4, 2), nhận được {pts.shape}")

    # Tính tổng x + y
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    # Tính hiệu y - x
    diff = np.diff(pts, axis=1).flatten()  # y - x
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def recover_missing_corner(
    corners: np.ndarray,
    visibilities: np.ndarray,
    conf_threshold: float = 0.4,
) -> Tuple[np.ndarray, bool, Optional[int]]:
    """
    Phục hồi 1 góc bị che dựa trên đặc tính hình học mặt phẳng bàn cờ:
    Trong hình bình hành/hình chữ nhật phẳng: TL + BR = TR + BL
    
    Args:
        corners: shape (4, 2) [TL, TR, BR, BL]
        visibilities: shape (4,) điểm tin cậy/visibility flag của từng góc
        conf_threshold: Ngưỡng xác định góc bị che/mất
        
    Returns:
        (recovered_corners, is_recovered, recovered_index)
    """
    corners = np.array(corners, dtype=np.float32).copy()
    valid_mask = visibilities >= conf_threshold
    num_valid = np.sum(valid_mask)

    if num_valid == 4:
        return corners, False, None

    if num_valid == 3:
        missing_idx = int(np.where(~valid_mask)[0][0])
        # Chỉ số: 0: TL, 1: TR, 2: BR, 3: BL
        if missing_idx == 0:  # TL = TR + BL - BR
            corners[0] = corners[1] + corners[3] - corners[2]
        elif missing_idx == 1:  # TR = TL + BR - BL
            corners[1] = corners[0] + corners[2] - corners[3]
        elif missing_idx == 2:  # BR = TR + BL - TL
            corners[2] = corners[1] + corners[3] - corners[0]
        elif missing_idx == 3:  # BL = TL + BR - TR
            corners[3] = corners[0] + corners[2] - corners[1]

        return corners, True, missing_idx

    # Nếu mất từ 2 góc trở lên thì không đủ thông tin hình học suy diễn
    return corners, False, None


class BoardCornerDetector:
    """
    Bộ phát hiện góc bàn cờ (Giai đoạn A) sử dụng YOLO-Pose kết hợp Geometric Fallback.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        corner_conf_threshold: float = 0.4,
        enable_recovery: bool = True,
        device: str = "cpu",
    ):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.corner_conf_threshold = corner_conf_threshold
        self.enable_recovery = enable_recovery
        self.device = device
        self.model = None

        if weights_path is not None:
            self._load_model(weights_path)

    def _load_model(self, weights_path: str):
        """Khởi tạo mô hình YOLO-Pose nếu đã có trọng số"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.model.to(self.device)
        except Exception as e:
            # Ghi nhận cảnh báo nếu chưa có file trọng số, cho phép hoạt động ở chế độ fallback
            self.model = None

    def detect_corners(self, image: np.ndarray) -> Optional[BoardCornersResult]:
        """
        Phát hiện 4 góc bàn cờ trên khung hình.
        
        Returns:
            BoardCornersResult hoặc None nếu không tìm thấy bàn cờ.
        """
        if self.model is not None:
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            if not results or len(results[0].keypoints) == 0:
                return None

            # Lấy keypoints của bàn cờ có confidence cao nhất
            keypoints_data = results[0].keypoints.data.cpu().numpy()[0]  # shape (4, 3) -> [x, y, conf]
            raw_pts = keypoints_data[:, :2]
            confidences = keypoints_data[:, 2]

            # Sắp xếp theo thứ tự TL, TR, BR, BL
            ordered_pts = order_corners_clockwise(raw_pts)

            if self.enable_recovery:
                final_corners, is_recovered, recovered_idx = recover_missing_corner(
                    ordered_pts, confidences, self.corner_conf_threshold
                )
            else:
                final_corners, is_recovered, recovered_idx = ordered_pts, False, None

            return BoardCornersResult(
                corners=final_corners,
                confidences=confidences,
                is_recovered=is_recovered,
                recovered_corner_index=recovered_idx,
            )

        return None
