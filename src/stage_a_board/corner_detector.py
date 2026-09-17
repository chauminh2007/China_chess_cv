"""
Phát hiện 4 góc bàn cờ (YOLO-Pose Corner Detector) và Thuật toán khôi phục góc bị che.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class BoardCornersResult:
    """Kết quả phát hiện 4 góc bàn cờ"""
    corners: np.ndarray             # shape (4, 2) theo thứ tự [TL, TR, BR, BL]
    confidences: np.ndarray         # shape (4,) confidence của từng góc
    is_recovered: bool              # True nếu có 1 góc được phục hồi từ hình học
    recovered_corner_index: Optional[int] = None  # Chỉ số góc (0..3) được phục hồi


def order_corners_clockwise_with_index(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sắp xếp 4 điểm bất kỳ theo thứ tự chuẩn: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    Trả về cả mảng chỉ số hoán vị (sort_index) để caller có thể áp cùng permutation
    lên các mảng liên kết (ví dụ: confidences) nhằm đảm bảo confidences[i] luôn
    tương ứng với corners[i] sau khi sắp xếp lại.

    Returns:
        ordered_pts  : np.ndarray shape (4, 2) — các điểm đã sắp xếp [TL, TR, BR, BL]
        sort_index   : np.ndarray shape (4,) dtype int — chỉ số nguyên bản tương ứng,
                       sao cho ordered_pts[i] == pts[sort_index[i]]
    """
    pts = np.array(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Kỳ vọng mảng (4, 2), nhận được {pts.shape}")

    # Tính tổng x + y để xác định TL (min) và BR (max)
    s = pts.sum(axis=1)
    tl_idx = int(np.argmin(s))
    br_idx = int(np.argmax(s))

    # Tính hiệu y - x để xác định TR (min) và BL (max)
    diff = np.diff(pts, axis=1).flatten()  # y - x
    tr_idx = int(np.argmin(diff))
    bl_idx = int(np.argmax(diff))

    sort_index = np.array([tl_idx, tr_idx, br_idx, bl_idx], dtype=np.intp)
    ordered_pts = pts[sort_index]
    return ordered_pts, sort_index


def order_corners_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 điểm bất kỳ theo thứ tự chuẩn: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    Lưu ý: Hàm này CHỈ trả về các điểm đã sắp xếp, không trả về sort_index.
    Nếu cần áp cùng permutation lên mảng liên kết (ví dụ confidences),
    hãy dùng order_corners_clockwise_with_index() thay thế.
    """
    ordered_pts, _ = order_corners_clockwise_with_index(pts)
    return ordered_pts


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

    # Nếu mất từ 2 góc trở lên thì không đủ thông tin hình học để suy diễn.
    # Trả về (corners_gốc, False, None) — caller cần kiểm tra num_valid để biết
    # đây là trường hợp "không đủ góc" chứ không phải "đủ 4 góc nhưng không cần phục hồi".
    num_missing = 4 - int(num_valid)
    logger.warning(
        "recover_missing_corner: %d góc bị che (< ngưỡng %.2f) — "
        "không đủ thông tin hình học để phục hồi. "
        "Pipeline nên bỏ frame này hoặc yêu cầu người dùng điều chỉnh camera.",
        num_missing,
        conf_threshold,
    )
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
            raw_confidences = keypoints_data[:, 2]

            # Sắp xếp theo thứ tự TL, TR, BR, BL.
            # QUAN TRỌNG: dùng order_corners_clockwise_with_index() để lấy sort_index,
            # sau đó áp CÙNG permutation lên confidences — đảm bảo confidences[i]
            # luôn tương ứng với corners[i] sau khi reorder.
            # Nếu chỉ reorder pts mà giữ nguyên confidences (thứ tự model output),
            # recover_missing_corner sẽ dùng sai confidence và có thể phục hồi
            # nhầm góc đang hiện rõ, bỏ qua góc thực sự bị che.
            ordered_pts, sort_index = order_corners_clockwise_with_index(raw_pts)
            ordered_confidences = raw_confidences[sort_index]

            if self.enable_recovery:
                final_corners, is_recovered, recovered_idx = recover_missing_corner(
                    ordered_pts, ordered_confidences, self.corner_conf_threshold
                )
            else:
                final_corners, is_recovered, recovered_idx = ordered_pts, False, None

            return BoardCornersResult(
                corners=final_corners,
                confidences=ordered_confidences,  # trả confidence đã reorder, khớp với corners
                is_recovered=is_recovered,
                recovered_corner_index=recovered_idx,
            )

        return None
