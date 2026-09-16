"""
Giai đoạn B: Định vị quân cờ trên ảnh đã chuẩn hóa.

Câu hỏi mà giai đoạn này trả lời: "Ở đâu có quân cờ?"
(KHÔNG hỏi quân đó là quân gì - đó là nhiệm vụ của Giai đoạn C)

Tại sao tách riêng Giai đoạn B:
    Vì ảnh đã được warp về góc nhìn thẳng đứng ở Giai đoạn A,
    kích thước quân cờ gần như cố định bất kể góc chụp ban đầu.
    Detector chỉ cần học 1 nhiệm vụ đơn giản: "hình tròn có quân cờ".
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2


@dataclass
class DetectedPieceBox:
    """
    Bounding box của 1 quân cờ phát hiện được trên canvas chuẩn hóa.

    Các trường:
        bbox          - Tọa độ (x1, y1, x2, y2) của bounding box
        center_x      - Tọa độ x của tâm quân cờ
        center_y      - Tọa độ y của tâm quân cờ
        confidence    - Độ tin cậy của detector (0.0 - 1.0)
        class_id      - Lớp detector (0 = quân cờ chung, hoặc 0/1 = đỏ/đen)
        cropped_patch - Ảnh crop của quân cờ, sẵn sàng đưa vào Giai đoạn C
    """
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center_x: float
    center_y: float
    confidence: float
    class_id: int = 0
    cropped_patch: Optional[np.ndarray] = None


class RectifiedPieceDetector:
    """
    Phát hiện vị trí quân cờ trên ảnh đã warp chuẩn hóa.

    Đặc điểm:
        - Đầu vào: Ảnh warped 720x800 (cố định kích thước, nhìn thẳng đứng)
        - Đầu ra: Danh sách DetectedPieceBox (tọa độ + ảnh crop từng quân)
        - Không phân loại quân là gì, chỉ tìm VỊ TRÍ
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.35,     # Bỏ qua detection có confidence < ngưỡng này
        iou_threshold: float = 0.45,       # Ngưỡng IoU cho NMS (loại box trùng nhau)
        crop_size: Tuple[int, int] = (64, 64),  # Kích thước ảnh patch sau khi crop
        crop_padding: int = 4,             # Số pixel padding thêm xung quanh khi crop
        device: str = "cpu",
    ):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.crop_size = crop_size
        self.crop_padding = crop_padding
        self.device = device
        self.model = None  # Model YOLO, None nếu chưa có trọng số

        if weights_path is not None:
            self._load_model(weights_path)

    def _load_model(self, weights_path: str):
        """Tải model YOLO từ file trọng số"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.model.to(self.device)
        except Exception as e:
            # Nếu không load được (file không tồn tại, lỗi,...) thì để model = None
            # Pipeline vẫn chạy được nhưng Giai đoạn B sẽ không phát hiện quân nào
            print(f"Cảnh báo: Không thể load model Stage B từ {weights_path}: {e}")
            self.model = None

    def detect(self, rectified_image: np.ndarray) -> List[DetectedPieceBox]:
        """
        Tìm tất cả quân cờ trong ảnh đã chuẩn hóa.

        Luồng xử lý:
          1. Chạy YOLO trên toàn bộ ảnh để lấy bounding boxes
          2. Với mỗi box: tính tọa độ tâm + crop patch vuông quanh quân
          3. Trả về danh sách DetectedPieceBox

        Args:
            rectified_image - Ảnh warped từ Giai đoạn A (numpy array BGR)

        Returns:
            Danh sách các quân cờ phát hiện được, mỗi quân kèm ảnh crop.
            Trả về list rỗng nếu không phát hiện được quân nào.
        """
        # Chưa có model -> không thể detect, trả về rỗng
        if self.model is None:
            return []

        # Chạy YOLO inference
        yolo_results = self.model(
            rectified_image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,  # Tắt log output của YOLO
        )

        # Trường hợp YOLO không tìm thấy gì
        if not yolo_results or len(yolo_results[0].boxes) == 0:
            return []

        # Lấy các tensor kết quả từ YOLO và chuyển về numpy
        raw_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()    # Shape (N, 4): x1,y1,x2,y2
        confidences = yolo_results[0].boxes.conf.cpu().numpy()  # Shape (N,)
        class_ids = yolo_results[0].boxes.cls.cpu().numpy().astype(int)  # Shape (N,)

        # Xây dựng danh sách kết quả
        detected_pieces: List[DetectedPieceBox] = []
        for (x1, y1, x2, y2), conf, cls_id in zip(raw_boxes, confidences, class_ids):

            # Tính tọa độ tâm của bounding box
            center_x = float((x1 + x2) / 2.0)
            center_y = float((y1 + y2) / 2.0)
            bbox = (float(x1), float(y1), float(x2), float(y2))

            # Crop ảnh vuông xung quanh quân cờ để đưa vào Giai đoạn C
            patch = self.crop_piece_patch(rectified_image, bbox)

            detected_pieces.append(DetectedPieceBox(
                bbox=bbox,
                center_x=center_x,
                center_y=center_y,
                confidence=float(conf),
                class_id=int(cls_id),
                cropped_patch=patch,
            ))

        return detected_pieces

    def crop_piece_patch(
        self,
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Cắt ảnh patch vuông xung quanh 1 quân cờ và resize về kích thước chuẩn.

        Tại sao phải crop vuông?
          Vì quân cờ Tướng có hình tròn, nên bounding box lý tưởng là hình vuông.
          Nếu crop hình chữ nhật rồi resize thành vuông sẽ gây méo ảnh.

        Cách hoạt động:
          1. Tính tâm (cx, cy) và bán kính half_side từ bounding box + padding
          2. Tính vùng crop [px1, py1, px2, py2] đảm bảo không vượt ra ngoài ảnh
          3. Nếu vùng crop không vuông (do sát mép ảnh), pad thêm bằng pixel đen
          4. Resize về target_size (mặc định 64x64)

        Args:
            image       - Ảnh nguồn (numpy BGR)
            bbox        - Bounding box (x1, y1, x2, y2)
            target_size - Kích thước đầu ra (mặc định dùng self.crop_size)

        Returns:
            Ảnh patch numpy shape (H, W, 3) đã resize
        """
        x1, y1, x2, y2 = bbox
        image_height, image_width = image.shape[:2]
        output_size = target_size or self.crop_size

        # Tính tâm và bán nửa cạnh vuông (lấy max của width/height + padding)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        half_width = (x2 - x1) / 2.0 + self.crop_padding
        half_height = (y2 - y1) / 2.0 + self.crop_padding
        half_side = max(half_width, half_height)  # Đảm bảo crop hình vuông

        # Tính tọa độ vùng crop, clamp về trong biên ảnh
        crop_x1 = int(max(0, np.floor(center_x - half_side)))
        crop_y1 = int(max(0, np.floor(center_y - half_side)))
        crop_x2 = int(min(image_width, np.ceil(center_x + half_side)))
        crop_y2 = int(min(image_height, np.ceil(center_y + half_side)))

        # Cắt vùng ảnh
        patch = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # Trả về ảnh đen nếu vùng crop bị rỗng (quân nằm hoàn toàn ngoài ảnh)
        if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
            return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

        # Nếu bị sát mép ảnh, patch có thể không vuông -> pad thêm pixel đen
        patch_h, patch_w = patch.shape[:2]
        if patch_h != patch_w:
            max_dim = max(patch_h, patch_w)
            square_patch = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
            # Căn giữa patch trong khung vuông
            offset_y = (max_dim - patch_h) // 2
            offset_x = (max_dim - patch_w) // 2
            square_patch[offset_y:offset_y + patch_h, offset_x:offset_x + patch_w] = patch
            patch = square_patch

        # Resize về kích thước đầu ra chuẩn
        resized_patch = cv2.resize(patch, output_size, interpolation=cv2.INTER_AREA)
        return resized_patch
