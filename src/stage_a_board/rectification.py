"""
Khử méo ống kính (Camera Undistort) và Tính biến đổi phối cảnh Homography.
Chuyển đổi phối cảnh bất kỳ về góc nhìn chuẩn hóa (Top-Down Canonical View).

Xử lý orientation:
    Bàn cờ Tướng có thể được chụp theo 2 hướng:
        - DỌC (portrait): chiều dài bàn cờ song song với trục Y ảnh   (9 cột x 10 hàng)
        - NGANG (landscape): chiều dài bàn cờ song song với trục X ảnh (10 cột x 9 hàng)
    Code tự động phát hiện và luôn warp về canvas DỌC chuẩn hóa.
"""

from typing import Optional, Tuple
import numpy as np
import cv2

from .canonical_grid import CanonicalBoardGrid


class CameraCalibrator:
    """
    Xử lý khử méo phi tuyến của ống kính máy ảnh (Lens Distortion Correction).
    """

    def __init__(
        self,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ):
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32) if camera_matrix is not None else None
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32) if dist_coeffs is not None else None

    @property
    def is_calibrated(self) -> bool:
        return self.camera_matrix is not None and self.dist_coeffs is not None

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Khử méo ảnh nếu đã có thông số calibrate, ngược lại trả về nguyên bản"""
        if not self.is_calibrated:
            return image
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)


def detect_board_orientation(src_corners: np.ndarray) -> str:
    """
    Phát hiện bàn cờ đang đặt DỌC hay NGANG dựa trên tọa độ 4 góc.

    Cách tính:
        Tính chiều rộng trung bình (cạnh trên + cạnh dưới) và
        chiều cao trung bình (cạnh trái + cạnh phải) của hình thang.
        - Nếu chiều cao > chiều rộng  -> bàn cờ DỌC  (portrait)
        - Nếu chiều rộng > chiều cao  -> bàn cờ NGANG (landscape)

    Args:
        src_corners - Mảng shape (4, 2) theo thứ tự [TL, TR, BR, BL]

    Returns:
        "portrait"  - bàn cờ dọc
        "landscape" - bàn cờ nằm ngang
    """
    tl, tr, br, bl = src_corners  # Top-Left, Top-Right, Bottom-Right, Bottom-Left

    # Chiều rộng: cạnh trên (TL->TR) và cạnh dưới (BL->BR), lấy trung bình
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    avg_width = (width_top + width_bottom) / 2.0

    # Chiều cao: cạnh trái (TL->BL) và cạnh phải (TR->BR), lấy trung bình
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    avg_height = (height_left + height_right) / 2.0

    if avg_height >= avg_width:
        return "portrait"   # Bàn cờ dọc: chiều cao >= chiều rộng
    else:
        return "landscape"  # Bàn cờ ngang: chiều rộng > chiều cao


def rotate_corners_for_landscape(src_corners: np.ndarray) -> np.ndarray:
    """
    Khi bàn cờ nằm ngang, xoay thứ tự 4 góc 90 độ ngược chiều kim đồng hồ.

    Lý do cần xoay:
        Canvas chuẩn hóa luôn là DỌC (portrait, 720x800).
        Nếu bàn nằm ngang mà không xoay góc trước khi tính Homography,
        bàn cờ sẽ bị warp sai hướng (9 cột thành hàng ngang thay vì dọc).

    Cách xoay (giả sử bàn nằm ngang theo chiều thuận kim đồng hồ 90 độ):
        Góc cũ [TL, TR, BR, BL] (bàn ngang) ứng với:
        Góc mới [BL, TL, TR, BR] (bàn dọc)

        Trực quan:
            Bàn ngang:            Xoay thành bàn dọc:
            TL ─── TR             BL ─── TL
            │       │      ->     │       │
            BL ─── BR             BR ─── TR

    Args:
        src_corners - Mảng (4, 2) theo thứ tự [TL, TR, BR, BL] của bàn ngang

    Returns:
        Mảng (4, 2) theo thứ tự [TL, TR, BR, BL] sau khi đã xoay
    """
    tl, tr, br, bl = src_corners
    # Sau khi xoay 90 độ ngược chiều kim đồng hồ:
    # BL cũ -> TL mới, TL cũ -> TR mới, TR cũ -> BR mới, BR cũ -> BL mới
    new_tl = bl
    new_tr = tl
    new_br = tr
    new_bl = br
    return np.array([new_tl, new_tr, new_br, new_bl], dtype=np.float32)


class HomographyRectifier:
    """
    Tính ma trận Homography H và warp ảnh về kích thước chuẩn hóa cố định.

    Tự động xử lý bàn cờ ngang (landscape) bằng cách xoay 4 góc trước
    khi tính Homography, đảm bảo kết quả warp luôn là bàn cờ DỌC.
    """

    def __init__(self, canonical_grid: Optional[CanonicalBoardGrid] = None):
        self.grid = canonical_grid or CanonicalBoardGrid()
        self.canonical_corners = self.grid.canonical_corners.copy()
        self.homography_matrix: Optional[np.ndarray] = None
        self.inv_homography_matrix: Optional[np.ndarray] = None
        self.src_corners: Optional[np.ndarray] = None
        self.detected_orientation: str = "portrait"  # Ghi lại để dùng khi debug

    def compute_homography(self, src_corners: np.ndarray) -> np.ndarray:
        """
        Tính ma trận Homography H từ 4 góc phát hiện được trên ảnh gốc.

        Luồng xử lý:
          1. Phát hiện bàn cờ đang đặt dọc hay ngang
          2. Nếu ngang: xoay thứ tự 4 góc 90 độ để khớp với canvas dọc
          3. Tính Homography từ 4 góc (đã điều chỉnh) -> 4 góc canonical

        Args:
            src_corners - Mảng shape (4, 2) theo thứ tự [TL, TR, BR, BL]

        Returns:
            H - Ma trận Homography (3, 3)
        """
        src = np.array(src_corners, dtype=np.float32)
        if src.shape != (4, 2):
            raise ValueError(f"Tọa độ 4 góc phải có shape (4, 2), nhận được {src.shape}")

        # BƯỚC 1: Phát hiện orientation (dọc/ngang)
        self.detected_orientation = detect_board_orientation(src)

        # BƯỚC 2: Nếu bàn cờ nằm ngang, xoay thứ tự góc để khớp với canvas dọc
        if self.detected_orientation == "landscape":
            src = rotate_corners_for_landscape(src)
            # (Không cần xoay ảnh - Homography sẽ tự lo việc biến đổi)

        # BƯỚC 3: Tính ma trận Homography
        # H biến đổi: src_corners (ảnh gốc) -> canonical_corners (canvas chuẩn hóa)
        self.src_corners = src
        self.homography_matrix = cv2.getPerspectiveTransform(src, self.canonical_corners)
        self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)

        return self.homography_matrix

    def warp(self, image: np.ndarray, border_mode: int = cv2.BORDER_CONSTANT, border_value: int = 0) -> np.ndarray:
        """
        Warp ảnh gốc về canvas chuẩn hóa (luôn là bàn cờ DỌC 720x800).

        Bất kể bàn cờ chụp dọc hay ngang, kết quả luôn là ảnh dọc
        nhờ bước xoay góc trong compute_homography().
        """
        if self.homography_matrix is None:
            raise RuntimeError("Chưa tính ma trận Homography! Hãy gọi compute_homography() trước.")

        warped = cv2.warpPerspective(
            image,
            self.homography_matrix,
            (self.grid.canvas_width, self.grid.canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
            borderValue=border_value,
        )
        return warped

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi tọa độ các điểm từ ảnh gốc sang canvas chuẩn hóa (Forward Transform).

        Args:
            points - shape (N, 2)
        Returns:
            transformed_points - shape (N, 2)
        """
        if self.homography_matrix is None:
            raise RuntimeError("Chưa tính ma trận Homography!")

        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.homography_matrix)
        return transformed.reshape(-1, 2)

    def inverse_transform_points(self, canonical_points: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi tọa độ các điểm từ canvas chuẩn hóa ngược về ảnh gốc (Inverse Transform).

        Args:
            canonical_points - shape (N, 2)
        Returns:
            original_points - shape (N, 2)
        """
        if self.inv_homography_matrix is None:
            raise RuntimeError("Chưa tính ma trận Homography!")

        pts = np.array(canonical_points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.inv_homography_matrix)
        return transformed.reshape(-1, 2)
