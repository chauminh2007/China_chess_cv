"""
Dataset và Data Augmentation cho Giai đoạn C (Piece Patch Classification).
Xử lý mất cân bằng dữ liệu (Class Imbalance) và biến đổi hình học (xoay đa hướng).
"""

from typing import List, Optional, Tuple, Union
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from ..post_processing.piece_definitions import PIECE_REGISTRY


class PiecePatchDataset(Dataset):
    """
    PyTorch Dataset chứa các ảnh crop của quân cờ và nhãn lớp (0..13).
    """

    def __init__(
        self,
        samples: List[Tuple[Union[str, np.ndarray, Image.Image], int]],
        transform: Optional[T.Compose] = None,
        is_train: bool = True,
    ):
        """
        Args:
            samples: Danh sách các tuple (đường dẫn ảnh hoặc ndarray, class_id)
            transform: Chuỗi biến đổi ảnh torchvision transforms
            is_train: Chế độ train (sử dụng augmentation) hoặc eval
        """
        self.samples = samples
        self.is_train = is_train
        self.transform = transform or self._default_transform(is_train)

    @classmethod
    def from_directory(cls, data_dir: str, is_train: bool = True) -> "PiecePatchDataset":
        """
        Tải dataset từ cấu trúc thư mục dạng ImageFolder: data_dir/{class_name}/*.png
        """
        from ..post_processing.piece_definitions import NAME_TO_CLASS_ID

        samples = []
        for class_name, class_id in NAME_TO_CLASS_ID.items():
            class_folder = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_folder):
                continue

            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for img_path in glob.glob(os.path.join(class_folder, ext)):
                    samples.append((img_path, class_id))

        return cls(samples=samples, is_train=is_train)

    def _default_transform(self, is_train: bool) -> T.Compose:
        """Tạo transform chuẩn cho quân cờ"""
        if is_train:
            return T.Compose([
                T.Resize((64, 64)),
                # Xoay ngẫu nhiên vì quân cờ trên thực tế có thể quay bất kỳ hướng nào
                T.RandomRotation(degrees=180),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return T.Compose([
                T.Resize((64, 64)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def compute_class_weights(self) -> torch.Tensor:
        """
        Tính trọng số nghịch đảo tần suất lớp (Inverse Class Frequency Weights)
        để truyền vào CrossEntropyLoss, giải quyết việc Tướng (1 quân) ít hơn Tốt (5 quân).
        """
        counts = np.zeros(14, dtype=np.float32)
        for _, class_id in self.samples:
            if 0 <= class_id < 14:
                counts[class_id] += 1.0

        total_samples = len(self.samples)
        num_classes = 14

        # Tránh chia cho 0 nếu lớp nào đó chưa có trong tập train
        counts = np.maximum(counts, 1.0)
        weights = total_samples / (num_classes * counts)

        # Chuẩn hóa về mean = 1
        weights = weights / np.mean(weights)
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item, label = self.samples[idx]

        if isinstance(item, str):
            image = Image.open(item).convert("RGB")
        elif isinstance(item, np.ndarray):
            # Giả định ảnh OpenCV (BGR) -> RGB
            if item.shape[2] == 3:
                rgb_arr = item[..., ::-1]
            else:
                rgb_arr = item
            image = Image.fromarray(rgb_arr)
        elif isinstance(item, Image.Image):
            image = item.convert("RGB")
        else:
            raise TypeError(f"Loại dữ liệu ảnh không hợp lệ: {type(item)}")

        tensor_img = self.transform(image)
        return tensor_img, label
