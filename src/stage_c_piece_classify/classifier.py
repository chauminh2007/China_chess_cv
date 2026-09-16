"""
Giai đoạn C - Inference: Phân loại quân cờ từ ảnh patch crop.

Câu hỏi mà giai đoạn này trả lời: "Quân này là quân gì?"
Đầu vào: Ảnh patch nhỏ (64x64) của 1 quân cờ đã crop từ Giai đoạn B.
Đầu ra: Tên quân + màu (1 trong 14 lớp: Xe đỏ, Xe đen, Mã đỏ, ... Tướng đỏ, Tướng đen).
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from .model import build_classifier_model
from ..post_processing.piece_definitions import (
    PIECE_REGISTRY,
    PieceInfo,
)


@dataclass
class ClassificationResult:
    """
    Kết quả phân loại của 1 quân cờ.

    Các trường:
        class_id      - Chỉ số lớp (0..13), tương ứng với 14 loại quân
        class_name    - Tên mã lớp (vd: "red_rook", "black_king")
        confidence    - Xác suất cao nhất (0.0 - 1.0)
        piece_info    - Thông tin đầy đủ của quân (tên, màu, ký tự FEN, ...)
        probabilities - Mảng xác suất cho cả 14 lớp (để debug nếu cần)
    """
    class_id: int
    class_name: str
    confidence: float
    piece_info: PieceInfo
    probabilities: np.ndarray


class PieceClassifier:
    """
    Bộ phân loại quân cờ - nhận ảnh patch nhỏ, trả về tên quân và màu.

    Luồng inference cho 1 ảnh:
        numpy (BGR, 64x64)
            -> đổi BGR->RGB -> PIL Image -> Resize -> ToTensor -> Normalize
            -> model.forward() -> logits (14 giá trị)
            -> softmax -> probabilities (14 xác suất, tổng = 1.0)
            -> argmax -> class_id tốt nhất
    """

    # Giá trị chuẩn hóa ImageNet (dùng vì backbone MobileNet/EfficientNet được pretrained trên ImageNet)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        weights_path: Optional[str] = None,
        backbone: str = "mobilenet_v3_small",
        device: str = "cpu",
        input_size: Tuple[int, int] = (64, 64),
    ):
        """
        Args:
            weights_path - Đường dẫn file .pt chứa trọng số đã huấn luyện.
                           Nếu None hoặc file không tồn tại -> dùng trọng số ngẫu nhiên.
            backbone     - Kiến trúc backbone ("mobilenet_v3_small" hoặc "custom_cnn")
            device       - "cpu" hoặc "cuda"
            input_size   - Kích thước ảnh đầu vào model (phải khớp với lúc train)
        """
        self.device = torch.device(device)
        self.input_size = input_size
        self.backbone = backbone
        self.weights_path = weights_path

        # Tải trọng số hoặc khởi tạo model mới với trọng số ngẫu nhiên
        if weights_path is not None and os.path.isfile(weights_path):
            self._load_weights(weights_path)
        else:
            # Chưa có trọng số -> tạo model trống (dùng để test cấu trúc)
            self.model = build_classifier_model(
                backbone=backbone,
                num_classes=14,
                pretrained=False,
            )

        self.model.to(self.device)
        self.model.eval()  # Chuyển về chế độ inference (tắt dropout, batch norm theo train mode)

        # Pipeline chuẩn hóa ảnh: giống hệt pipeline dùng khi train để kết quả nhất quán
        self.transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),  # Đổi [0,255] -> [0.0,1.0] và chuyển về (C,H,W)
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def _load_weights(self, weights_path: str):
        """
        Tải trọng số từ file checkpoint.

        File checkpoint có thể có 2 định dạng:
          (a) Dict với "model_state_dict" và "backbone" key (lưu bởi trainer.py)
          (b) Trực tiếp là state_dict (format đơn giản)

        Tự động đọc backbone type từ checkpoint để tránh mismatch shape.
        """
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            backbone_to_use = self.backbone
            state_dict = checkpoint

            # Kiểm tra xem checkpoint có phải định dạng dict đầy đủ không
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                # Đọc backbone type từ checkpoint để đảm bảo model đúng kiến trúc
                backbone_to_use = checkpoint.get("backbone", self.backbone)

            # Tạo model với đúng backbone rồi nạp trọng số
            self.model = build_classifier_model(
                backbone=backbone_to_use,
                num_classes=14,
                pretrained=False,
            )
            self.model.load_state_dict(state_dict)
            self.backbone = backbone_to_use
            print(f"Đã nạp thành công trọng số Stage C ({backbone_to_use}) từ: {weights_path}")

        except Exception as e:
            print(f"Cảnh báo: Không thể nạp trọng số Stage C từ {weights_path}: {e}")
            print("Dùng model với trọng số ngẫu nhiên (kết quả phân loại sẽ không chính xác).")
            self.model = build_classifier_model(
                backbone=self.backbone,
                num_classes=14,
                pretrained=False,
            )

    def _convert_patch_to_tensor(self, patch_bgr: np.ndarray) -> torch.Tensor:
        """
        Chuyển đổi ảnh numpy (BGR, từ OpenCV) sang Tensor chuẩn hóa cho model.

        Các bước:
          1. Đổi kênh màu: BGR (OpenCV) -> RGB (PIL/PyTorch)
          2. Bọc thành PIL Image
          3. Áp dụng transform: resize -> ToTensor -> normalize
        """
        # OpenCV dùng BGR, PIL/PyTorch dùng RGB -> cần đảo kênh màu
        if patch_bgr.ndim == 3 and patch_bgr.shape[2] == 3:
            patch_rgb = patch_bgr[..., ::-1]  # Đổi BGR -> RGB bằng đảo ngược trục kênh
        else:
            patch_rgb = patch_bgr  # Trường hợp ảnh xám, giữ nguyên

        pil_image = Image.fromarray(patch_rgb)
        tensor = self.transform(pil_image)
        return tensor

    @torch.no_grad()  # Tắt tính gradient để tiết kiệm bộ nhớ khi inference
    def predict_patch(self, patch: np.ndarray) -> ClassificationResult:
        """
        Phân loại 1 quân cờ đơn lẻ.

        Luồng:
          patch (numpy) -> tensor -> model -> logits -> softmax -> xác suất -> kết quả
        """
        # Thêm dimension batch (model cần input shape [batch, C, H, W])
        tensor = self._convert_patch_to_tensor(patch).unsqueeze(0).to(self.device)

        # Forward pass qua model
        logits = self.model(tensor)  # Shape: (1, 14)

        # Softmax để đổi logits thành xác suất (tổng = 1.0)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]  # Shape: (14,)

        # Lấy class có xác suất cao nhất
        best_class_id = int(np.argmax(probabilities))
        best_confidence = float(probabilities[best_class_id])
        piece_info = PIECE_REGISTRY[best_class_id]

        return ClassificationResult(
            class_id=best_class_id,
            class_name=piece_info.name_code,
            confidence=best_confidence,
            piece_info=piece_info,
            probabilities=probabilities,
        )

    @torch.no_grad()
    def predict_batch(self, patches: List[np.ndarray]) -> List[ClassificationResult]:
        """
        Phân loại nhiều quân cờ cùng lúc (batch inference).

        Tại sao dùng batch thay vì gọi predict_patch từng cái?
          GPU/CPU hiệu quả hơn nhiều khi xử lý nhiều ảnh cùng lúc.
          Batch inference nhanh hơn ~N lần so với N lần gọi đơn lẻ.

        Luồng:
          [patch_1, patch_2, ..., patch_N]
              -> N tensors -> stack thành batch (N, C, H, W)
              -> model -> logits (N, 14)
              -> softmax -> N bộ xác suất
              -> argmax từng dòng -> N kết quả
        """
        if not patches:
            return []

        # Chuyển từng patch sang tensor và gộp thành 1 batch
        individual_tensors = [self._convert_patch_to_tensor(p) for p in patches]
        batch_tensor = torch.stack(individual_tensors, dim=0).to(self.device)  # Shape: (N, C, H, W)

        # Chạy model 1 lần cho toàn bộ batch
        batch_logits = self.model(batch_tensor)  # Shape: (N, 14)
        batch_probabilities = F.softmax(batch_logits, dim=1).cpu().numpy()  # Shape: (N, 14)

        # Chuyển từng hàng xác suất thành ClassificationResult
        results: List[ClassificationResult] = []
        for probs in batch_probabilities:
            best_class_id = int(np.argmax(probs))
            best_confidence = float(probs[best_class_id])
            piece_info = PIECE_REGISTRY[best_class_id]

            results.append(ClassificationResult(
                class_id=best_class_id,
                class_name=piece_info.name_code,
                confidence=best_confidence,
                piece_info=piece_info,
                probabilities=probs,
            ))

        return results
