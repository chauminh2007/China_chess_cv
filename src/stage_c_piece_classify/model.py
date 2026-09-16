"""
Kiến trúc mô hình phân loại chi tiết 14 loại quân cờ tướng (MobileNetV3 / Lightweight CNN).
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class CustomLightweightCNN(nn.Module):
    """
    Mạng CNN nhỏ gọn thiết kế riêng cho việc phân loại chữ Hán trên quân cờ (patch 64x64).
    Tốc độ cực nhanh trên CPU mà vẫn đảm bảo độ chính xác cao.
    """

    def __init__(self, num_classes: int = 14, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 8x8 -> 8x8 (Residual-style conv)
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        out = self.classifier(pooled)
        return out


def build_classifier_model(
    backbone: str = "mobilenet_v3_small",
    num_classes: int = 14,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function khởi tạo mạng phân loại quân cờ.
    """
    if backbone == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model
    elif backbone == "custom_cnn":
        return CustomLightweightCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Backbone không hỗ trợ: {backbone}. Hãy chọn 'mobilenet_v3_small' hoặc 'custom_cnn'.")
