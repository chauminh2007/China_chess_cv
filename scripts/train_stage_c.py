"""
CLI Script: Huấn luyện mô hình phân loại 14 loại quân cờ (Stage C Trainer).
Cách dùng:
    python scripts/train_stage_c.py --data_dir data/synthetic/stage_c_classifier --epochs 30 --batch_size 64
"""

import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage_c_piece_classify.dataset import PiecePatchDataset
from src.stage_c_piece_classify.model import build_classifier_model
from src.stage_c_piece_classify.trainer import PieceClassifierTrainer


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại 14 loại quân cờ.")
    parser.add_argument("--data_dir", type=str, default="data/synthetic/stage_c_classifier", help="Thư mục chứa tập patch train/val")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "custom_cnn"])
    parser.add_argument("--epochs", type=int, default=30, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Tốc độ học (Learning Rate)")
    parser.add_argument("--loss", type=str, default="focal_loss", choices=["focal_loss", "weighted_ce", "ce"])
    parser.add_argument("--save_path", type=str, default="models/stage_c_classifier.pt", help="Đường dẫn lưu trọng số tốt nhất")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.isdir(train_dir):
        print(f"Lỗi: Không tìm thấy thư mục {train_dir}. Hãy chạy scripts/generate_synthetic_data.py trước!")
        return

    print("Đang nạp dataset...")
    train_dataset = PiecePatchDataset.from_directory(train_dir, is_train=True)
    val_dataset = PiecePatchDataset.from_directory(val_dir, is_train=False)

    print(f"Số mẫu Train: {len(train_dataset)}, Số mẫu Val: {len(val_dataset)}")
    if len(train_dataset) == 0:
        print("Lỗi: Dataset trống!")
        return

    # Tính trọng số lớp giải quyết mất cân bằng
    class_weights = train_dataset.compute_class_weights() if args.loss in ("focal_loss", "weighted_ce") else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Khởi tạo mô hình
    model = build_classifier_model(backbone=args.backbone, num_classes=14, pretrained=True)

    trainer = PieceClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        loss_type=args.loss,
        class_weights=class_weights,
        device=args.device,
    )

    print(f"Bắt đầu huấn luyện Stage C ({args.backbone}) trên thiết bị {args.device}...")
    # Lưu backbone trong trainer
    trainer.backbone_name = args.backbone
    trainer.fit(epochs=args.epochs, save_path=args.save_path)


if __name__ == "__main__":
    main()
