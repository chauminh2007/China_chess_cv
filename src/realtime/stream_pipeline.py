"""
Bộ điều phối luồng thời gian thực (Real-time Stream Pipeline).
Tối ưu hóa tài nguyên: Stage A chạy lúc setup/drift, Stage B+C chạy theo cơ chế hướng sự kiện (Event-Driven).
"""

from typing import Optional, Tuple, Dict, Any
import time
import numpy as np
import cv2

from .change_detector import BoardChangeDetector, BoardMotionState
from ..stage_a_board.corner_detector import BoardCornerDetector, BoardCornersResult
from ..stage_a_board.rectification import CameraCalibrator, HomographyRectifier
from ..stage_a_board.canonical_grid import CanonicalBoardGrid
from ..stage_b_piece_detect.piece_detector import RectifiedPieceDetector
from ..stage_c_piece_classify.classifier import PieceClassifier
from ..post_processing.grid_snapper import GridSnapper, RawPieceDetection
from ..post_processing.board_state import BoardState


class XiangqiStreamProcessor:
    """
    Xử lý luồng camera / video thời gian thực cho hệ thống Cờ Tướng thị giác máy tính.
    """

    def __init__(
        self,
        corner_detector: BoardCornerDetector,
        piece_detector: RectifiedPieceDetector,
        piece_classifier: PieceClassifier,
        canonical_grid: Optional[CanonicalBoardGrid] = None,
        camera_calibrator: Optional[CameraCalibrator] = None,
        motion_threshold: float = 15.0,
        stability_frames: int = 15,
        max_snap_radius: float = 38.0,
        stage_a_interval_sec: float = 30.0,
    ):
        self.corner_detector = corner_detector
        self.piece_detector = piece_detector
        self.piece_classifier = piece_classifier
        self.canonical_grid = canonical_grid or CanonicalBoardGrid()
        self.calibrator = camera_calibrator or CameraCalibrator()
        self.rectifier = HomographyRectifier(self.canonical_grid)
        self.snapper = GridSnapper(self.canonical_grid.grid_points_2d, max_snap_radius_px=max_snap_radius)
        self.board_state = BoardState(enforce_piece_limits=True)

        self.change_detector = BoardChangeDetector(
            motion_threshold=motion_threshold,
            stability_frames_required=stability_frames,
        )

        self.stage_a_interval_sec = stage_a_interval_sec
        self.last_stage_a_time: float = 0.0
        self.is_board_localized: bool = False
        self.last_corners_result: Optional[BoardCornersResult] = None
        self.last_fen: str = ""
        self.last_warped_board: Optional[np.ndarray] = None

    def initialize_board(self, undistorted_frame: np.ndarray) -> bool:
        """
        Khởi tạo Giai đoạn A: Tìm 4 góc bàn và tính ma trận Homography.

        Args:
            undistorted_frame: Ảnh đã qua undistort từ CameraCalibrator.
                               KHÔNG gọi undistort() bên trong hàm này để tránh áp 2 lần
                               (undistort không phải phép biến đổi idempotent).
        """
        corners_res = self.corner_detector.detect_corners(undistorted_frame)

        if corners_res is not None:
            self.rectifier.compute_homography(corners_res.corners)
            self.last_corners_result = corners_res
            self.is_board_localized = True
            self.change_detector.reset_baseline(undistorted_frame)
            return True
        return False

    def process_frame(
        self,
        frame: np.ndarray,
        force_detect: bool = False,
    ) -> Dict[str, Any]:
        """
        Xử lý 1 khung hình từ luồng video.
        
        Returns:
            Dictionary chứa các thông số:
            - state: Trạng thái chuyển động hiện tại
            - motion_score: Điểm chuyển động
            - is_updated: True nếu vừa hoàn thành 1 lần quét B+C mới
            - fen: Chuỗi FEN mới nhất
            - board_state: Đối tượng BoardState
            - debug_vis: Khung hình trực quan hóa kết quả
        """
        undistorted = self.calibrator.undistort(frame)
        now = time.time()

        # 1. Kiểm tra nếu chưa định vị bàn cờ hoặc đến chu kỳ kiểm tra drift
        if not self.is_board_localized or (now - self.last_stage_a_time > self.stage_a_interval_sec):
            # Cập nhật timestamp TRƯỚC KHI thử — đảm bảo throttle luôn hoạt động
            # dù detect_corners thành công hay thất bại (tay che bàn, nhiễu, v.v.)
            # Nếu không làm vậy: khi detect thất bại, last_stage_a_time không đổi
            # → frame kế tiếp điều kiện > 30s vẫn đúng → Stage A bị gọi mọi frame.
            self.last_stage_a_time = now
            self.initialize_board(undistorted)
            if not self.is_board_localized:
                return {
                    "status": "AWAITING_BOARD_SETUP",
                    "state": BoardMotionState.IDLE,
                    "motion_score": 0.0,
                    "is_updated": False,
                    "fen": self.last_fen,
                    "board_state": self.board_state,
                }

        # 2. Chạy Change Detection
        motion_state, motion_score = self.change_detector.process_frame(undistorted)
        is_updated = False

        # 3. Kích hoạt Giai đoạn B + C khi có tín hiệu TRIGGER_DETECT hoặc force_detect
        if motion_state == BoardMotionState.TRIGGER_DETECT or force_detect:
            # Warp sang canvas chuẩn hóa
            warped = self.rectifier.warp(undistorted)
            self.last_warped_board = warped.copy()

            # Giai đoạn B: Định vị quân cờ
            piece_boxes = self.piece_detector.detect(warped)

            if piece_boxes:
                # Giai đoạn C: Cắt patch và phân loại 14 lớp
                patches = [box.cropped_patch for box in piece_boxes]
                class_results = self.piece_classifier.predict_batch(patches)

                # Chuẩn bị dữ liệu cho Hậu xử lý (Snap)
                raw_detections = []
                for box, res in zip(piece_boxes, class_results):
                    raw_detections.append(
                        RawPieceDetection(
                            center_x=box.center_x,
                            center_y=box.center_y,
                            bbox=box.bbox,
                            class_id=res.class_id,
                            class_name=res.class_name,
                            confidence=res.confidence,
                            det_confidence=box.confidence,
                        )
                    )

                # Snap vào lưới và áp dụng luật
                snapped_pieces = self.snapper.snap_detections(raw_detections)
                warnings = self.board_state.update_from_snapped_pieces(snapped_pieces)
                self.last_fen = self.board_state.to_fen()
                is_updated = True
            else:
                self.board_state.clear()
                self.last_fen = self.board_state.to_fen()
                is_updated = True

        return {
            "status": "OK",
            "state": motion_state,
            "motion_score": motion_score,
            "is_updated": is_updated,
            "fen": self.last_fen,
            "board_state": self.board_state,
            "corners": self.last_corners_result,
            "warped_image": self.last_warped_board,
        }

    def render_overlay(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Vẽ thông tin debug và kết quả trực tiếp lên khung hình"""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Vẽ 4 góc bàn cờ nếu có
        corners_res: Optional[BoardCornersResult] = result.get("corners")
        if corners_res is not None and corners_res.corners is not None:
            pts = corners_res.corners.astype(int)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            for idx, pt in enumerate(pts):
                color = (0, 165, 255) if corners_res.is_recovered and idx == corners_res.recovered_corner_index else (0, 255, 0)
                cv2.circle(vis, tuple(pt), 6, color, -1)
                cv2.putText(vis, f"C{idx}", (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Vẽ thanh trạng thái
        state_str = str(result.get("state", ""))
        motion = result.get("motion_score", 0.0)
        fen_str = result.get("fen", "")

        status_color = (0, 255, 0) if "TRIGGER" in state_str else ((0, 165, 255) if "MOVING" in state_str else (255, 255, 255))
        cv2.putText(vis, f"State: {state_str} (Motion: {motion:.1f}%)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(vis, f"FEN: {fen_str[:40]}...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return vis
