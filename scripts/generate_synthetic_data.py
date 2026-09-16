"""
CLI Script: Sinh dữ liệu tổng hợp tự động cho Stage A, Stage B và Stage C.
Cách dùng:
    python scripts/generate_synthetic_data.py --num_train 200 --num_val 40 --output_dir data/synthetic
"""

import argparse
import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthetic.dataset_generator import SyntheticDatasetExporter
from src.synthetic.board_renderer import SyntheticXiangqiRenderer


def main():
    parser = argparse.ArgumentParser(description="Sinh dữ liệu tổng hợp cho hệ thống Nhận diện Cờ Tướng.")
    parser.add_argument("--num_train", type=int, default=100, help="Số lượng bàn cờ tập train (mặc định: 100)")
    parser.add_argument("--num_val", type=int, default=20, help="Số lượng bàn cờ tập validation (mặc định: 20)")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Thư mục lưu dataset")
    parser.add_argument("--occlusion_prob", type=float, default=0.25, help="Xác suất che khuất 1 góc ngẫu nhiên")
    parser.add_argument("--font_path", type=str, default=None, help="Đường dẫn file font chữ Hán tuỳ chỉnh")
    args = parser.parse_args()

    renderer = SyntheticXiangqiRenderer(font_path=args.font_path)
    exporter = SyntheticDatasetExporter(output_root=args.output_dir, renderer=renderer)

    exporter.generate_full_dataset(
        num_train_boards=args.num_train,
        num_val_boards=args.num_val,
        occlusion_prob=args.occlusion_prob,
    )


if __name__ == "__main__":
    main()
