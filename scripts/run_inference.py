"""
CLI Script: Chạy inference nhận diện bàn cờ tướng từ ảnh tĩnh, video hoặc webcam.
Cách dùng:
    # Nhận diện ảnh đơn:
    python scripts/run_inference.py --image path/to/image.jpg --output_vis output.jpg

    # Chạy luồng webcam thời gian thực:
    python scripts/run_inference.py --webcam 0
"""

import argparse
import sys
import os
import cv2
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import XiangqiVisionPipeline
from src.realtime.stream_pipeline import XiangqiStreamProcessor


def run_single_image(pipeline: XiangqiVisionPipeline, image_path: str, output_vis: str):
    """Chạy nhận diện trên 1 ảnh tĩnh"""
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file ảnh {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Lỗi: Không thể đọc file ảnh.")
        return

    print(f"Đang xử lý ảnh: {image_path}...")
    res = pipeline.process_image(image)

    if not res.is_success:
        print("Nhận diện thất bại! Cảnh báo:", res.warnings)
        return

    print("\n" + "=" * 50)
    print("KẾT QUẢ NHẬN DIỆN BÀN CỜ TƯỚNG:")
    print("=" * 50)
    print(res.board_state.to_ascii_display())
    print("\nCHUỖI FEN XUẤT RA:")
    print(res.fen)
    print("=" * 50)

    if res.warnings:
        print("\nCảnh báo/Điều chỉnh luật:", res.warnings)

    if output_vis and "annotated_canvas" in res.debug_visualizations:
        cv2.imwrite(output_vis, res.debug_visualizations["annotated_canvas"])
        print(f"Đã lưu ảnh trực quan hóa vào: {output_vis}")


def run_camera_stream(pipeline: XiangqiVisionPipeline, cam_id: int):
    """Chạy nhận diện trên luồng camera thời gian thực với Change Detection & Debounce"""
    processor = XiangqiStreamProcessor(
        corner_detector=pipeline.corner_detector,
        piece_detector=pipeline.piece_detector,
        piece_classifier=pipeline.piece_classifier,
        canonical_grid=pipeline.canonical_grid,
        camera_calibrator=pipeline.calibrator,
    )

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở camera {cam_id}")
        return

    print("Bắt đầu luồng camera. Nhấn 'q' để thoát, 'r' để reset định vị bàn cờ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = processor.process_frame(frame)
        vis_frame = processor.render_overlay(frame, result)

        if result.get("is_updated"):
            print(f"\n[EVENT] Bàn cờ vừa thay đổi và ổn định! FEN mới: {result['fen']}")
            print(result["board_state"].to_ascii_display())

        cv2.imshow("Xiangqi CV Realtime Monitor", vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            processor.is_board_localized = False
            print("Đang định vị lại bàn cờ...")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Chạy nhận diện Cờ Tướng.")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn file ảnh đầu vào")
    parser.add_argument("--output_vis", type=str, default="output_annotated.jpg", help="Đường dẫn lưu ảnh kết quả")
    parser.add_argument("--webcam", type=int, default=None, help="Index webcam (ví dụ: 0)")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml", help="File cấu hình pipeline")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    pipeline = XiangqiVisionPipeline(config_path=args.config, device=args.device)

    if args.image:
        run_single_image(pipeline, args.image, args.output_vis)
    elif args.webcam is not None:
        run_camera_stream(pipeline, args.webcam)
    else:
        print("Vui lòng chỉ định --image hoặc --webcam. Xem --help để biết thêm chi tiết.")


if __name__ == "__main__":
    main()
