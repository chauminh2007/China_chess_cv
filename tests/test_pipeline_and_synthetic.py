"""
Unit tests kiểm tra Bộ sinh dữ liệu tổng hợp, Máy trạng thái phát hiện chuyển động (Change Detector) và Mạng phân loại.
"""

import unittest
import numpy as np
import torch

from src.synthetic.board_renderer import SyntheticXiangqiRenderer, SyntheticBoardOutput
from src.realtime.change_detector import BoardChangeDetector, BoardMotionState
from src.stage_c_piece_classify.model import CustomLightweightCNN, build_classifier_model
from src.stage_c_piece_classify.trainer import FocalLoss


class TestPipelineAndSynthetic(unittest.TestCase):

    def test_synthetic_renderer_generation(self):
        """Kiểm tra sinh dữ liệu tổng hợp 2D/3D và nhãn ground-truth"""
        renderer = SyntheticXiangqiRenderer()
        out: SyntheticBoardOutput = renderer.generate_random_board(occlude_one_corner=True)

        self.assertEqual(out.canonical_image.shape, (800, 720, 3))
        self.assertEqual(out.projected_image.shape, (768, 1024, 3))
        self.assertEqual(out.corners_projected.shape, (4, 2))
        self.assertEqual(len(out.corners_visibility), 4)

        # Kiểm tra 1 góc bị che (vis == 0.0)
        self.assertEqual(np.sum(out.corners_visibility == 0.0), 1)

        # Bàn cờ ban đầu có đúng 32 quân
        self.assertEqual(len(out.pieces), 32)

    def test_change_detector_state_machine(self):
        """Kiểm tra máy trạng thái phát hiện chuyển động và debounce"""
        detector = BoardChangeDetector(motion_threshold=10.0, stability_frames_required=5)

        # Khung hình nền tĩnh
        frame_a = np.zeros((480, 640, 3), dtype=np.uint8)
        state, motion = detector.process_frame(frame_a)
        self.assertEqual(state, BoardMotionState.IDLE)

        # Khung hình có chuyển động lớn (tay người đưa vào)
        frame_moving = np.full((480, 640, 3), 180, dtype=np.uint8)
        state, motion = detector.process_frame(frame_moving)
        self.assertEqual(state, BoardMotionState.MOVING)
        self.assertGreater(motion, 10.0)

        # Trạng thái tĩnh liên tiếp sau khi tay rút đi (Debounce trong 5 frames)
        for i in range(4):
            state, _ = detector.process_frame(frame_moving)
            self.assertEqual(state, BoardMotionState.SETTLING)

        # Frame thứ 5 đạt ngưỡng ổn định -> Kích hoạt TRIGGER_DETECT
        state, _ = detector.process_frame(frame_moving)
        self.assertEqual(state, BoardMotionState.TRIGGER_DETECT)

        # Frame tiếp theo sau khi trigger -> Quay lại IDLE
        state, _ = detector.process_frame(frame_moving)
        self.assertEqual(state, BoardMotionState.IDLE)

    def test_classifier_model_forward_and_focal_loss(self):
        """Kiểm tra forward pass mạng CNN và tính Focal Loss"""
        model = CustomLightweightCNN(num_classes=14)
        dummy_input = torch.randn(4, 3, 64, 64)
        output = model(dummy_input)

        self.assertEqual(output.shape, (4, 14))

        # Kiểm tra Focal Loss
        targets = torch.tensor([0, 6, 7, 13], dtype=torch.long)
        criterion = FocalLoss(gamma=2.0)
        loss = criterion(output, targets)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
