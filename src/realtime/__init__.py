"""
Package Thời gian thực: Phát hiện biến đổi & Điều phối luồng xử lý (Real-time Change Detection & Stream Processing).
"""

from .change_detector import BoardChangeDetector, BoardMotionState
from .stream_pipeline import XiangqiStreamProcessor

__all__ = [
    "BoardChangeDetector",
    "BoardMotionState",
    "XiangqiStreamProcessor",
]
