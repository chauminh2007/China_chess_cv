"""
Unit tests cho XiangqiStreamProcessor — bộ điều phối luồng thời gian thực.

Tập trung kiểm tra logic điều phối state/timing vốn không được cover bởi các test khác:
    - test_geometry.py       → hình học góc, Homography
    - test_post_processing.py → snap lưới, FEN
    - test_pipeline_and_synthetic.py → pipeline tĩnh (ảnh đơn)

Tại sao cần test riêng:
    Logic điều phối real-time (throttle 30s, state máy, undistort flow) dễ sinh bug
    tinh tế nhất (ví dụ: last_stage_a_time không cập nhật khi detect thất bại).
    Mock toàn bộ ML model để test chạy nhanh không cần GPU/weights.
"""

import time
import unittest
from unittest.mock import MagicMock
import numpy as np

from src.realtime.stream_pipeline import XiangqiStreamProcessor
from src.stage_a_board.corner_detector import BoardCornersResult
from src.realtime.change_detector import BoardMotionState


def _make_dummy_corners() -> BoardCornersResult:
    """Tạo BoardCornersResult giả với 4 góc hợp lệ (hình chữ nhật đứng)."""
    return BoardCornersResult(
        corners=np.array(
            [[100.0, 100.0], [500.0, 100.0], [500.0, 700.0], [100.0, 700.0]],
            dtype=np.float32,
        ),
        confidences=np.ones(4, dtype=np.float32),
        is_recovered=False,
        recovered_corner_index=None,
    )


def _make_blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Tạo ảnh BGR trống để làm đầu vào test."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_processor(detect_return=None) -> XiangqiStreamProcessor:
    """
    Tạo XiangqiStreamProcessor với toàn bộ ML component được mock.

    Args:
        detect_return: Giá trị trả về của corner_detector.detect_corners().
                       None = detect thất bại; BoardCornersResult = detect thành công.
    """
    corner_detector = MagicMock()
    corner_detector.detect_corners.return_value = detect_return

    piece_detector = MagicMock()
    piece_detector.detect.return_value = []

    piece_classifier = MagicMock()
    piece_classifier.predict_batch.return_value = []

    # Mock CameraCalibrator: undistort trả về ảnh nguyên bản (chưa calibrate)
    calibrator = MagicMock()
    calibrator.undistort.side_effect = lambda img: img  # Identity function

    # Mock HomographyRectifier: warp trả về ảnh trống
    rectifier_mock = MagicMock()
    rectifier_mock.warp.return_value = _make_blank_frame()

    # Mock ChangeDetector: luôn báo IDLE (không kích hoạt B+C)
    change_detector = MagicMock()
    change_detector.process_frame.return_value = (BoardMotionState.IDLE, 0.0)
    change_detector.reset_baseline.return_value = None

    proc = XiangqiStreamProcessor(
        corner_detector=corner_detector,
        piece_detector=piece_detector,
        piece_classifier=piece_classifier,
        camera_calibrator=calibrator,
        stage_a_interval_sec=30.0,
    )
    # Inject mock trực tiếp vào instance
    proc.rectifier = rectifier_mock
    proc.change_detector = change_detector

    return proc


class TestThrottleTiming(unittest.TestCase):
    """
    Kiểm tra logic throttle 30s của Stage A.

    Bug đã sửa (Bug 1):
        last_stage_a_time phải được cập nhật NGAY KHI THỬ gọi Stage A,
        bất kể detect_corners thành công hay thất bại.
        Nếu chỉ cập nhật khi thành công: mỗi frame trong suốt thời gian
        tay che bàn sẽ gọi Stage A (model nặng nhất) ở mọi frame.
    """

    def test_throttle_updates_on_detect_failure(self):
        """
        Khi detect thất bại (trả None), last_stage_a_time vẫn phải được cập nhật.
        Frame kế tiếp không được gọi lại Stage A trong vòng 30s tiếp theo.
        """
        # detect trả None → simulate tay che bàn
        proc = _make_processor(detect_return=None)
        frame = _make_blank_frame()

        # Đặt is_board_localized=True và last_stage_a_time rất cũ để trigger recheck
        proc.is_board_localized = True
        proc.last_stage_a_time = time.time() - 40.0  # Đã qua 40s > 30s interval

        before_call = time.time()
        proc.process_frame(frame)

        # Sau khi thử (dù thất bại), last_stage_a_time phải được cập nhật
        self.assertGreaterEqual(
            proc.last_stage_a_time,
            before_call - 0.1,  # Margin nhỏ cho clock jitter
            "last_stage_a_time phải được cập nhật ngay cả khi detect_corners trả None",
        )

    def test_throttle_prevents_repeated_stage_a_calls_during_occlusion(self):
        """
        Khi tay che bàn liên tục (detect luôn trả None), Stage A chỉ được gọi
        1 lần mỗi interval 30s, KHÔNG phải mọi frame.
        """
        proc = _make_processor(detect_return=None)
        frame = _make_blank_frame()

        proc.is_board_localized = True
        proc.last_stage_a_time = time.time() - 40.0  # Trigger recheck ngay

        # Frame 1: thử detect → thất bại → throttle reset
        proc.process_frame(frame)
        stage_a_call_count_after_frame1 = proc.corner_detector.detect_corners.call_count
        self.assertEqual(stage_a_call_count_after_frame1, 1, "Frame 1 phải gọi Stage A đúng 1 lần")

        # Frame 2, 3, 4: interval 30s chưa hết → Stage A KHÔNG được gọi thêm
        for _ in range(3):
            proc.process_frame(frame)

        total_calls = proc.corner_detector.detect_corners.call_count
        self.assertEqual(
            total_calls,
            1,
            f"Stage A bị gọi {total_calls} lần trong 4 frame kế tiếp — "
            "throttle không hoạt động khi detect thất bại!",
        )

    def test_throttle_updates_on_detect_success(self):
        """Khi detect thành công, last_stage_a_time cũng phải được cập nhật (regression)."""
        proc = _make_processor(detect_return=_make_dummy_corners())
        frame = _make_blank_frame()

        proc.is_board_localized = False  # Chưa localize → luôn thử
        before_call = time.time()
        proc.process_frame(frame)

        self.assertGreaterEqual(
            proc.last_stage_a_time,
            before_call - 0.1,
            "last_stage_a_time phải được cập nhật khi detect thành công",
        )


class TestDoubleUndistort(unittest.TestCase):
    """
    Kiểm tra rằng ảnh không bị undistort 2 lần.

    Bug đã sửa (Bug 2):
        process_frame() gọi undistort(frame) → undistorted
        rồi truyền undistorted vào initialize_board()
        Bên trong initialize_board() cũ lại gọi undistort() thêm lần nữa
        → undistort 2 lần liên tiếp → ảnh méo theo hướng ngược lại.
    """

    def test_initialize_board_does_not_call_undistort_internally(self):
        """
        initialize_board() không được gọi calibrator.undistort() bên trong.
        undistort phải được thực hiện DUY NHẤT bởi process_frame().
        """
        proc = _make_processor(detect_return=_make_dummy_corners())
        frame = _make_blank_frame()

        # Gọi trực tiếp initialize_board với ảnh đã undistort
        undistorted = proc.calibrator.undistort(frame)
        undistort_call_count_before = proc.calibrator.undistort.call_count

        proc.initialize_board(undistorted)

        undistort_call_count_after = proc.calibrator.undistort.call_count

        self.assertEqual(
            undistort_call_count_before,
            undistort_call_count_after,
            "initialize_board() không được gọi calibrator.undistort() — "
            "chỉ process_frame() mới được gọi undistort một lần duy nhất.",
        )

    def test_process_frame_calls_undistort_exactly_once_per_frame(self):
        """
        Mỗi lần gọi process_frame(), undistort phải được gọi đúng 1 lần —
        dù có trigger Stage A hay không.
        """
        proc = _make_processor(detect_return=_make_dummy_corners())
        frame = _make_blank_frame()

        proc.is_board_localized = False  # Trigger Stage A
        proc.process_frame(frame)

        self.assertEqual(
            proc.calibrator.undistort.call_count,
            1,
            "process_frame() phải gọi undistort đúng 1 lần mỗi frame",
        )

    def test_undistort_call_count_over_multiple_frames(self):
        """
        Qua N frame, undistort phải được gọi đúng N lần (không nhân đôi do Stage A).
        """
        proc = _make_processor(detect_return=None)
        frame = _make_blank_frame()

        proc.is_board_localized = True
        proc.last_stage_a_time = time.time() - 40.0  # Trigger Stage A ở frame đầu

        n_frames = 5
        for _ in range(n_frames):
            proc.process_frame(frame)

        self.assertEqual(
            proc.calibrator.undistort.call_count,
            n_frames,
            f"Sau {n_frames} frame, undistort phải được gọi đúng {n_frames} lần",
        )


class TestBoardLocalizationState(unittest.TestCase):
    """
    Kiểm tra máy trạng thái định vị bàn cờ (AWAITING_BOARD_SETUP vs OK).
    """

    def test_awaiting_board_setup_when_never_localized(self):
        """Khi chưa từng detect được bàn, status phải là AWAITING_BOARD_SETUP."""
        proc = _make_processor(detect_return=None)
        frame = _make_blank_frame()

        proc.is_board_localized = False
        result = proc.process_frame(frame)

        self.assertEqual(result["status"], "AWAITING_BOARD_SETUP")

    def test_continues_with_old_homography_when_recheck_fails(self):
        """
        Khi bàn đã được localize (is_board_localized=True) và periodic recheck
        thất bại (tay che bàn), hệ thống phải tiếp tục với homography cũ (status='OK').
        """
        proc = _make_processor(detect_return=None)
        frame = _make_blank_frame()

        # Giả lập đã localize trước đó
        proc.is_board_localized = True
        proc.last_stage_a_time = time.time() - 40.0  # Trigger recheck

        result = proc.process_frame(frame)

        # Dù recheck thất bại, hệ thống vẫn tiếp tục (không về AWAITING_BOARD_SETUP)
        self.assertEqual(
            result["status"],
            "OK",
            "Hệ thống phải dùng homography cũ và tiếp tục khi recheck thất bại "
            "trong khi bàn đã được localize",
        )

    def test_status_ok_after_successful_localization(self):
        """Sau khi detect thành công, status phải là OK."""
        proc = _make_processor(detect_return=_make_dummy_corners())
        frame = _make_blank_frame()

        proc.is_board_localized = False
        result = proc.process_frame(frame)

        self.assertEqual(result["status"], "OK")
        self.assertTrue(proc.is_board_localized)


if __name__ == "__main__":
    unittest.main()
