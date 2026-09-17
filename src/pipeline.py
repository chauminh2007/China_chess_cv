"""
Pipeline hoàn chỉnh tích hợp Giai đoạn A -> B -> C -> Hậu xử lý -> FEN.

Luồng xử lý tổng thể:
    ảnh gốc
        -> [Giai đoạn A] Tìm 4 góc bàn cờ, Warp về góc nhìn thẳng đứng
        -> [Giai đoạn B] Detect vị trí các quân cờ trên ảnh đã chuẩn hóa
        -> [Giai đoạn C] Cắt từng quân và phân loại vào 14 lớp (7 loại * 2 màu)
        -> [Hậu xử lý]  Snap vào lưới 9x10, áp dụng ràng buộc luật cờ
        -> Xuất chuỗi FEN
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import yaml
import os
import cv2
import numpy as np

from .stage_a_board.canonical_grid import CanonicalBoardGrid
from .stage_a_board.rectification import CameraCalibrator, HomographyRectifier
from .stage_a_board.corner_detector import BoardCornerDetector, BoardCornersResult
from .stage_b_piece_detect.piece_detector import RectifiedPieceDetector, DetectedPieceBox
from .stage_c_piece_classify.classifier import PieceClassifier, ClassificationResult
from .post_processing.grid_snapper import GridSnapper, RawPieceDetection, SnappedPiece
from .post_processing.board_state import BoardState
from .post_processing.piece_definitions import PieceColor


@dataclass
class PipelineResult:
    """
    Kết quả hoàn chỉnh sau khi xử lý 1 ảnh qua toàn bộ pipeline 3 giai đoạn.

    Các trường:
        is_success       - True nếu pipeline chạy thành công đến cuối
        fen              - Chuỗi FEN biểu diễn trạng thái bàn cờ
        board_state      - Đối tượng trạng thái bàn cờ chi tiết
        warped_board     - Ảnh bàn cờ sau khi warp về góc nhìn thẳng đứng
        raw_corners      - 4 góc bàn cờ đã phát hiện
        detected_pieces  - Danh sách quân cờ đã được snap vào lưới
        warnings         - Danh sách cảnh báo (vi phạm luật, ...)
        debug_visualizations - Dict các ảnh debug để kiểm tra từng bước
    """
    is_success: bool
    fen: str
    board_state: BoardState
    warped_board: Optional[np.ndarray]
    raw_corners: Optional[BoardCornersResult]
    detected_pieces: List[SnappedPiece]
    warnings: List[str]
    debug_visualizations: Dict[str, np.ndarray]


class XiangqiVisionPipeline:
    """
    Hệ thống thị giác máy tính nhận diện Cờ Tướng theo kiến trúc 3 giai đoạn độc lập.

    Lý do tách thành 3 giai đoạn:
        - Giai đoạn A (định vị bàn cờ): Loại bỏ biến thiên góc chụp TRƯỚC khi nhận diện quân.
          Đây là nguyên nhân chính khiến YOLO học vẹt khi ném thẳng ảnh vào.
        - Giai đoạn B (detect quân): Sau khi warp, kích thước quân gần như cố định,
          giúp detector hoạt động ổn định bất kể góc chụp.
        - Giai đoạn C (phân loại quân): Chỉ nhận ảnh crop nhỏ của từng quân,
          tập trung học hình dạng/màu sắc, không bị nhiễu bởi bối cảnh bàn cờ.
    """

    def __init__(
        self,
        config_path: str = "configs/pipeline_config.yaml",
        device: str = "cpu",
    ):
        self.config_path = config_path
        self.device = device

        # Đọc toàn bộ cấu hình từ file YAML
        self.configs = self._load_all_configs(config_path)

        # Khởi tạo các thành phần cho từng giai đoạn
        self._init_stage_a()
        self._init_stage_b()
        self._init_stage_c()
        self._init_post_processing()

    # =========================================================================
    # KHỞI TẠO CÁC THÀNH PHẦN
    # =========================================================================

    def _init_stage_a(self):
        """Khởi tạo các thành phần của Giai đoạn A: Định vị bàn cờ"""
        stage_a_cfg = self.configs.get("stage_a", {})

        # Lưới chuẩn hóa 9x10 (định nghĩa tọa độ pixel chuẩn cho mỗi giao điểm)
        geom_cfg = stage_a_cfg.get("geometry", {})
        self.canonical_grid = CanonicalBoardGrid(
            canvas_width=geom_cfg.get("canvas_width", 720),
            canvas_height=geom_cfg.get("canvas_height", 800),
            margin_x=geom_cfg.get("margin_x", 40),
            margin_y=geom_cfg.get("margin_y", 40),
        )

        # Bộ khử méo ống kính (lens undistortion)
        # Nếu không có thông số calibrate thì bỏ qua bước này
        calib_cfg = stage_a_cfg.get("camera_calibration", {})
        self.calibrator = CameraCalibrator(
            camera_matrix=calib_cfg.get("camera_matrix"),
            dist_coeffs=calib_cfg.get("dist_coeffs"),
        )

        # Bộ tính Homography và warp ảnh về góc nhìn thẳng đứng (top-down)
        self.rectifier = HomographyRectifier(self.canonical_grid)

        # Bộ phát hiện 4 góc bàn cờ bằng YOLO-Pose
        occ_cfg = stage_a_cfg.get("occlusion_recovery", {})
        model_a_cfg = stage_a_cfg.get("model", {})
        self.corner_detector = BoardCornerDetector(
            weights_path=model_a_cfg.get("weights_path"),
            conf_threshold=model_a_cfg.get("conf_threshold", 0.5),
            corner_conf_threshold=occ_cfg.get("corner_conf_threshold", 0.4),
            enable_recovery=occ_cfg.get("enable_recovery", True),
            device=self.device,
        )

    def _init_stage_b(self):
        """Khởi tạo các thành phần của Giai đoạn B: Phát hiện vị trí quân cờ"""
        stage_b_cfg = self.configs.get("stage_b", {})
        model_b_cfg = stage_b_cfg.get("model", {})
        det_params = stage_b_cfg.get("detection_params", {})

        self.piece_detector = RectifiedPieceDetector(
            weights_path=model_b_cfg.get("weights_path"),
            conf_threshold=model_b_cfg.get("conf_threshold", 0.35),
            crop_size=tuple(det_params.get("target_crop_size", [64, 64])),
            crop_padding=det_params.get("crop_padding", 4),
            device=self.device,
        )

    def _init_stage_c(self):
        """Khởi tạo các thành phần của Giai đoạn C: Phân loại quân cờ"""
        stage_c_cfg = self.configs.get("stage_c", {})
        model_c_cfg = stage_c_cfg.get("model", {})

        self.piece_classifier = PieceClassifier(
            weights_path=model_c_cfg.get("weights_path"),
            backbone=model_c_cfg.get("backbone", "mobilenet_v3_small"),
            device=self.device,
        )

    def _init_post_processing(self):
        """Khởi tạo các thành phần Hậu xử lý: Snap lưới và ràng buộc luật cờ"""
        post_cfg = self.configs.get("pipeline", {}).get("post_processing", {})
        snap_cfg = post_cfg.get("grid_snapping", {})

        # Snap tọa độ pixel về giao điểm lưới gần nhất trên bàn 9x10
        self.snapper = GridSnapper(
            canonical_grid_points=self.canonical_grid.grid_points_2d,
            max_snap_radius_px=snap_cfg.get("max_snap_radius_px", 38.0),
            conflict_resolution=snap_cfg.get("conflict_resolution", "highest_conf"),
        )

        # Kiểm tra và áp dụng ràng buộc luật (số lượng quân tối đa mỗi loại)
        rule_cfg = post_cfg.get("rule_constraints", {})
        self.board_state = BoardState(
            enforce_piece_limits=rule_cfg.get("enforce_max_counts", True)
        )

    def _load_all_configs(self, main_config_path: str) -> Dict[str, Any]:
        """
        Đọc file cấu hình pipeline chính và các file cấu hình con (stage_a, b, c).

        Cấu trúc file cấu hình:
            pipeline_config.yaml  <- tham chiếu đến 3 file con bên dưới
                stage_a_config: path/to/stage_a.yaml
                stage_b_config: path/to/stage_b.yaml
                stage_c_config: path/to/stage_c.yaml
        """
        configs = {}

        if not os.path.exists(main_config_path):
            # Không có file config thì dùng giá trị mặc định trong từng __init__
            return {}

        with open(main_config_path, "r", encoding="utf-8") as f:
            pipeline_cfg = yaml.safe_load(f) or {}
        configs["pipeline"] = pipeline_cfg

        # Đọc file config của từng giai đoạn
        stage_files = {
            "stage_a": "stage_a_config",
            "stage_b": "stage_b_config",
            "stage_c": "stage_c_config",
        }
        for stage_key, config_key in stage_files.items():
            sub_path = pipeline_cfg.get(config_key)
            if sub_path and os.path.exists(sub_path):
                with open(sub_path, "r", encoding="utf-8") as f:
                    configs[stage_key] = yaml.safe_load(f) or {}
            else:
                configs[stage_key] = {}

        return configs

    # =========================================================================
    # XỬ LÝ CHÍNH
    # =========================================================================

    def process_image(
        self,
        image: np.ndarray,
        manual_corners: Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """
        Nhận diện toàn bộ bàn cờ từ 1 ảnh đầu vào.

        Args:
            image          - Ảnh BGR (NumPy array) từ camera hoặc file
            manual_corners - (Tuỳ chọn) Mảng shape (4, 2) tọa độ 4 góc bàn cờ
                             nếu muốn bỏ qua bước tự động phát hiện góc.

        Returns:
            PipelineResult chứa FEN, danh sách quân, và ảnh debug.
        """
        accumulated_warnings: List[str] = []

        # ----- BƯỚC 1: Khử méo ống kính -----
        # Nếu camera không có thông số calibrate thì ảnh trả về nguyên bản
        undistorted_image = self.calibrator.undistort(image)

        # ----- BƯỚC 2 (GIAI ĐOẠN A): Tìm 4 góc bàn cờ -----
        corners_result = self._run_stage_a_find_corners(undistorted_image, manual_corners)

        # Nếu không tìm được 4 góc -> không thể tiếp tục
        if corners_result is None:
            return self._make_failure_result("Giai đoạn A thất bại: Không thể tìm thấy 4 góc bàn cờ.")

        # Kiểm tra trường hợp ≥2 góc bị che: recover_missing_corner chỉ xử lý được đúng 1 góc.
        # Nếu có từ 2 góc trở lên dưới ngưỡng confidence và chưa được phục hồi
        # (is_recovered=False nhưng vẫn có góc confidence thấp) -> drop frame.
        # Tránh tiếp tục với 4 góc mà một số tọa độ không đáng tin, gây warp sai hoàn toàn.
        if not corners_result.is_recovered:
            occ_cfg = self.configs.get("stage_a", {}).get("occlusion_recovery", {})
            conf_threshold = occ_cfg.get(
                "corner_conf_threshold",
                self.corner_detector.corner_conf_threshold,
            )
            num_occluded = int(np.sum(corners_result.confidences < conf_threshold))
            if num_occluded >= 2:
                return self._make_failure_result(
                    f"Giai đoạn A thất bại: {num_occluded} góc bị che (confidence < {conf_threshold:.2f}) "
                    "và không thể phục hồi hình học (cần ≥3 góc rõ). "
                    "Vui lòng điều chỉnh góc camera hoặc loại bỏ vật che khuất."
                )

        # ----- BƯỚC 3 (GIAI ĐOẠN A): Warp ảnh về góc nhìn thẳng đứng -----
        # Sau bước này, ảnh luôn có kích thước cố định bất kể góc chụp ban đầu
        warped_board = self._run_stage_a_warp(undistorted_image, corners_result)

        # ----- BƯỚC 4 (GIAI ĐOẠN B): Phát hiện vị trí các quân cờ -----
        # Detect bounding box của tất cả quân trên ảnh đã chuẩn hóa
        detected_boxes: List[DetectedPieceBox] = self.piece_detector.detect(warped_board)

        # ----- BƯỚC 5 (GIAI ĐOẠN C): Phân loại từng quân cờ -----
        # Với mỗi bounding box, cắt patch và dự đoán là quân gì (14 lớp)
        raw_detections: List[RawPieceDetection] = self._run_stage_c_classify_all(
            warped_board, detected_boxes
        )

        # ----- BƯỚC 6: Snap vào lưới 9x10 và áp dụng ràng buộc luật cờ -----
        snapped_pieces, rule_warnings = self._run_post_processing(raw_detections)
        accumulated_warnings.extend(rule_warnings)

        # ----- BƯỚC 7: Xuất chuỗi FEN -----
        fen_string = self.board_state.to_fen()

        # ----- BƯỚC 8: Tạo ảnh trực quan hóa để debug -----
        debug_images = self._create_debug_visualizations(warped_board, snapped_pieces)

        return PipelineResult(
            is_success=True,
            fen=fen_string,
            board_state=self.board_state,
            warped_board=warped_board,
            raw_corners=corners_result,
            detected_pieces=snapped_pieces,
            warnings=accumulated_warnings,
            debug_visualizations=debug_images,
        )

    # =========================================================================
    # CÁC HÀM CON — MỖI HÀM ĐẢM NHIỆM 1 BƯỚC RÕ RÀNG
    # =========================================================================

    def _run_stage_a_find_corners(
        self,
        image: np.ndarray,
        manual_corners: Optional[np.ndarray],
    ) -> Optional[BoardCornersResult]:
        """
        Giai đoạn A - Bước 1: Tìm 4 góc bàn cờ.

        Có 2 cách:
          (a) Dùng góc thủ công nếu người dùng cung cấp manual_corners.
          (b) Dùng YOLO-Pose để tự phát hiện 4 góc, có thể phục hồi góc bị che.

        Kết quả: BoardCornersResult gồm tọa độ 4 góc theo thứ tự [TL, TR, BR, BL].
        """
        if manual_corners is not None:
            # Người dùng cung cấp góc thủ công -> sắp xếp lại theo thứ tự chuẩn
            from .stage_a_board.corner_detector import order_corners_clockwise
            ordered = order_corners_clockwise(manual_corners)
            return BoardCornersResult(
                corners=ordered,
                confidences=np.ones(4, dtype=np.float32),  # Độ tin cậy = 1.0 vì góc thủ công
                is_recovered=False,
            )
        else:
            # Tự động phát hiện góc bằng YOLO-Pose
            return self.corner_detector.detect_corners(image)

    def _run_stage_a_warp(
        self,
        image: np.ndarray,
        corners_result: BoardCornersResult,
    ) -> np.ndarray:
        """
        Giai đoạn A - Bước 2: Warp (biến đổi phối cảnh) ảnh về góc nhìn thẳng đứng chuẩn.

        Cách hoạt động:
          1. Tính ma trận Homography H từ 4 góc phát hiện được (nguồn)
             đến 4 góc của canvas chuẩn hóa cố định (đích).
          2. Áp dụng cv2.warpPerspective để biến đổi toàn bộ ảnh.

        Sau bước này:
          - Ảnh đầu ra luôn có kích thước cố định (vd: 720x800 pixel)
          - Bàn cờ luôn nhìn thẳng từ trên xuống, không bị nghiêng
          - Kích thước quân cờ gần như nhất quán giữa các ảnh
        """
        self.rectifier.compute_homography(corners_result.corners)
        warped_image = self.rectifier.warp(image)
        return warped_image

    def _run_stage_c_classify_all(
        self,
        warped_board: np.ndarray,
        detected_boxes: List[DetectedPieceBox],
    ) -> List[RawPieceDetection]:
        """
        Giai đoạn C: Phân loại tất cả quân cờ đã phát hiện trong 1 lần gọi batch.

        Cách hoạt động:
          1. Lấy danh sách ảnh patch (đã crop sẵn bởi Giai đoạn B) từ detected_boxes.
          2. Gọi predict_batch() để phân loại toàn bộ trong 1 lần (nhanh hơn gọi từng cái).
          3. Ghép kết quả phân loại (class, confidence) với tọa độ từ Giai đoạn B.

        Returns:
            Danh sách RawPieceDetection kết hợp tọa độ (từ B) và loại quân (từ C).
        """
        if not detected_boxes:
            return []  # Không có quân nào được phát hiện

        # Lấy các ảnh patch đã crop sẵn từ Giai đoạn B
        all_patches = [box.cropped_patch for box in detected_boxes]

        # Phân loại tất cả cùng lúc (batch inference, nhanh hơn từng ảnh)
        classification_results: List[ClassificationResult] = (
            self.piece_classifier.predict_batch(all_patches)
        )

        # Ghép tọa độ từ Giai đoạn B với kết quả phân loại từ Giai đoạn C
        combined_detections: List[RawPieceDetection] = []
        for box, cls_result in zip(detected_boxes, classification_results):
            combined_detections.append(
                RawPieceDetection(
                    center_x=box.center_x,
                    center_y=box.center_y,
                    bbox=box.bbox,
                    class_id=cls_result.class_id,
                    class_name=cls_result.class_name,
                    confidence=cls_result.confidence,
                    det_confidence=box.confidence,  # Confidence của detector (Giai đoạn B)
                )
            )

        return combined_detections

    def _run_post_processing(
        self,
        raw_detections: List[RawPieceDetection],
    ) -> Tuple[List[SnappedPiece], List[str]]:
        """
        Hậu xử lý: Snap tọa độ pixel vào lưới 9x10 và kiểm tra luật cờ.

        Bước 1 - Grid Snapping:
          Tọa độ pixel (x, y) từ Giai đoạn B/C không khớp chính xác với giao điểm lưới
          do nhiễu detection. Snap tìm giao điểm gần nhất trong lưới 9x10 cho mỗi quân.
          Nếu 2 quân cùng snap vào 1 ô -> giữ quân có confidence cao hơn.

        Bước 2 - Rule Constraints:
          Kiểm tra các ràng buộc luật cờ Tướng (vd: mỗi bên chỉ có 1 Tướng).
          Trả về danh sách cảnh báo nếu vi phạm.

        Returns:
            (danh sách quân đã snap, danh sách cảnh báo vi phạm luật)
        """
        snapped_pieces = self.snapper.snap_detections(raw_detections)
        rule_warnings = self.board_state.update_from_snapped_pieces(snapped_pieces)
        return snapped_pieces, rule_warnings

    def _create_debug_visualizations(
        self,
        warped_board: np.ndarray,
        snapped_pieces: List[SnappedPiece],
    ) -> Dict[str, np.ndarray]:
        """
        Tạo các ảnh trực quan hóa để debug và kiểm tra kết quả.

        Output:
            "warped_canvas"    - Ảnh bàn cờ đã warp (chưa vẽ gì thêm)
            "annotated_canvas" - Ảnh bàn cờ + lưới + quân cờ được đánh dấu
        """
        # Vẽ lưới lên ảnh để xem các giao điểm 9x10
        annotated_image = self.canonical_grid.draw_grid_overlay(warped_board)

        # Vẽ từng quân lên đúng vị trí trong lưới
        for piece in snapped_pieces:
            # Lấy tọa độ pixel tương ứng với ô (col, row) trong lưới
            pixel_x, pixel_y = self.canonical_grid.get_pixel_coord(piece.col, piece.row)

            # Màu đỏ cho quân đỏ, đen cho quân đen
            piece_color_bgr = (0, 0, 255) if piece.piece_info.color == PieceColor.RED else (0, 0, 0)

            # Vẽ vòng tròn trắng làm nền, rồi vẽ viền màu
            cv2.circle(annotated_image, (int(pixel_x), int(pixel_y)), 18, (255, 255, 255), -1)
            cv2.circle(annotated_image, (int(pixel_x), int(pixel_y)), 18, piece_color_bgr, 2)

            # Ghi ký tự FEN của quân lên giữa vòng tròn
            cv2.putText(
                annotated_image,
                piece.piece_info.fen_char,
                (int(pixel_x) - 7, int(pixel_y) + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                piece_color_bgr,
                2,
            )

        return {
            "warped_canvas": warped_board,
            "annotated_canvas": annotated_image,
        }

    def _make_failure_result(self, reason: str) -> PipelineResult:
        """
        Trả về PipelineResult thất bại khi pipeline không thể hoàn thành.
        Tách ra thành hàm riêng để tránh lặp code ở nhiều chỗ.
        """
        return PipelineResult(
            is_success=False,
            fen="",
            board_state=self.board_state,
            warped_board=None,
            raw_corners=None,
            detected_pieces=[],
            warnings=[reason],
            debug_visualizations={},
        )
