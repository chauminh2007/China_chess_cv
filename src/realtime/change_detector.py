"""
Phát hiện chuyển động và Ổn định bàn cờ (Change Detection & Debounce State Machine).
Giúp hệ thống không chạy B+C liên tục khi tay người đang di chuyển quân cờ,
chỉ kích hoạt nhận diện khi bàn cờ đã hoàn toàn đứng yên.
"""

from enum import Enum
from typing import Optional, Tuple
import numpy as np
import cv2


class BoardMotionState(str, Enum):
    IDLE = "IDLE"                        # Bàn cờ ổn định, chưa có thay đổi
    MOVING = "MOVING"                    # Đang có chuyển động (tay người/quân cờ di chuyển)
    SETTLING = "SETTLING"                # Chuyển động vừa dứt, đang đếm số frame ổn định (Debounce)
    TRIGGER_DETECT = "TRIGGER_DETECT"    # Đã ổn định sau thay đổi -> Kích hoạt Giai đoạn B+C!


class BoardChangeDetector:
    """
    Phát hiện biến đổi giữa các khung hình liên tiếp và theo dõi trạng thái ổn định.
    """

    def __init__(
        self,
        motion_threshold: float = 15.0,
        stability_frames_required: int = 15,
        diff_downsample_size: Tuple[int, int] = (320, 240),
        blur_kernel_size: int = 21,
    ):
        """
        Args:
            motion_threshold: Ngưỡng phần trăm điểm ảnh biến động để tính là đang chuyển động.
            stability_frames_required: Số frame liên tiếp dưới ngưỡng chuyển động trước khi trigger B+C.
            diff_downsample_size: Kích thước thu nhỏ để tính diff nhanh.
            blur_kernel_size: Kích thước Gaussian Blur để lọc nhiễu hạt camera.
        """
        self.motion_threshold = motion_threshold
        self.stability_frames_required = stability_frames_required
        self.diff_downsample_size = diff_downsample_size
        self.blur_kernel_size = blur_kernel_size

        self.last_stable_gray: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None

        self.state: BoardMotionState = BoardMotionState.IDLE
        self.stable_frames_count: int = 0
        self.current_motion_score: float = 0.0

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Thu nhỏ và làm mờ khung hình để tính sai phân ổn định"""
        resized = cv2.resize(frame, self.diff_downsample_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        return blurred

    def process_frame(self, frame: np.ndarray) -> Tuple[BoardMotionState, float]:
        """
        Xử lý từng khung hình và cập nhật máy trạng thái (State Machine).
        
        Returns:
            (trạng thái hiện tại, điểm số chuyển động motion_score)
        """
        gray = self._preprocess_frame(frame)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.last_stable_gray = gray
            self.state = BoardMotionState.IDLE
            return self.state, 0.0

        # Tính sai phân tuyệt đối giữa 2 khung hình liên tiếp
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Tính tỷ lệ pixel có chuyển động (%)
        motion_pixel_count = np.count_nonzero(thresh)
        total_pixels = thresh.size
        motion_score = (motion_pixel_count / total_pixels) * 100.0
        self.current_motion_score = motion_score
        self.prev_gray = gray

        is_moving_now = motion_score >= self.motion_threshold

        # Cập nhật State Machine
        if is_moving_now:
            # Phát hiện đang có tay người hoặc chuyển động lớn
            self.state = BoardMotionState.MOVING
            self.stable_frames_count = 0
        else:
            # Khung hình hiện tại tĩnh
            if self.state == BoardMotionState.MOVING or self.state == BoardMotionState.SETTLING:
                self.stable_frames_count += 1
                if self.stable_frames_count >= self.stability_frames_required:
                    # Đã ổn định đủ lâu sau khi di chuyển -> Kích hoạt nhận diện B+C!
                    self.state = BoardMotionState.TRIGGER_DETECT
                    self.stable_frames_count = 0
                    self.last_stable_gray = gray
                else:
                    self.state = BoardMotionState.SETTLING
            elif self.state == BoardMotionState.TRIGGER_DETECT:
                # Sau khi trigger 1 frame thì quay về trạng thái IDLE
                self.state = BoardMotionState.IDLE

        return self.state, motion_score

    def reset_baseline(self, frame: np.ndarray):
        """Đặt lại khung hình cơ sở khi bàn cờ mới được định vị"""
        gray = self._preprocess_frame(frame)
        self.last_stable_gray = gray
        self.prev_gray = gray
        self.state = BoardMotionState.IDLE
        self.stable_frames_count = 0
