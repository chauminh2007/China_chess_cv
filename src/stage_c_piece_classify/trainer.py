"""
Module huấn luyện mô hình phân loại quân cờ (Stage C Trainer).

Hỗ trợ 2 loại hàm mất mát:
    1. Focal Loss      - Tập trung vào các mẫu khó, giảm ảnh hưởng của mẫu dễ
    2. Weighted Cross-Entropy - Đơn giản hơn, phù hợp khi dữ liệu mất cân bằng nhẹ
"""

from typing import Optional, Dict, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class FocalLoss(nn.Module):
    """
    Focal Loss - Hàm mất mát tập trung vào mẫu khó.

    Vấn đề Focal Loss giải quyết:
        Trong tập dữ liệu mất cân bằng (vd: 1000 ảnh xe, 50 ảnh tướng),
        Cross Entropy thông thường bị chi phối bởi class nhiều mẫu.
        Model học "đoán đại lớp phổ biến" thay vì học thực sự.

    Ý tưởng:
        Thêm hệ số (1 - p_t)^gamma vào CE Loss để:
        - Giảm đóng góp của mẫu DỄ (p_t cao, model đã đoán đúng) -> (1-p_t)^gamma nhỏ
        - Giữ nguyên đóng góp của mẫu KHÓ (p_t thấp, model đoán sai) -> (1-p_t)^gamma lớn

    Công thức:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        - p_t    : xác suất model dự đoán đúng nhãn thực tế
        - gamma  : hệ số điều chỉnh (gamma=0 -> CE thông thường, gamma=2 là phổ biến)
        - alpha_t: trọng số lớp (tuỳ chọn, để ưu tiên các lớp ít mẫu hơn)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,  # Tensor (num_classes,) trọng số từng lớp
        gamma: float = 2.0,                    # Hệ số focal, thường dùng 2.0
        reduction: str = "mean",               # "mean" hoặc "sum" hoặc "none"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Tính Focal Loss cho 1 batch.

        Args:
            logits  - Output thô của model, shape (batch_size, num_classes)
            targets - Nhãn thực tế (integer), shape (batch_size,)

        Returns:
            Giá trị scalar loss
        """

        # BƯỚC 1: Tính CE Loss để lấy -ln(p_t) cho từng mẫu
        # reduction="none" để giữ loss từng mẫu riêng lẻ (không lấy mean ngay)
        ce_loss_per_sample = F.cross_entropy(logits, targets, reduction="none")
        # ce_loss_per_sample = -ln(p_t), shape (batch_size,)

        # BƯỚC 2: Tính ngược lại p_t từ ce_loss
        # Vì ce_loss = -ln(p_t) => p_t = e^(-ce_loss)
        predicted_prob_of_correct_class = torch.exp(-ce_loss_per_sample)
        # predicted_prob_of_correct_class = p_t, shape (batch_size,)

        # BƯỚC 3: Tính hệ số focal (1 - p_t)^gamma
        # - Với mẫu dễ: p_t ~ 1.0 => (1-p_t)^2 ~ 0.0 => weight nhỏ (gần 0)
        # - Với mẫu khó: p_t ~ 0.1 => (1-p_t)^2 ~ 0.81 => weight lớn (gần 1)
        focal_weight_per_sample = (1.0 - predicted_prob_of_correct_class) ** self.gamma

        # BƯỚC 4: Áp hệ số focal vào CE loss
        focal_loss_per_sample = focal_weight_per_sample * ce_loss_per_sample

        # BƯỚC 5: Nhân trọng số lớp alpha_t (nếu có)
        # alpha_t là phần tử của tensor alpha tại vị trí target[i]
        if self.alpha is not None:
            # Lấy trọng số tương ứng với nhãn của từng mẫu trong batch
            alpha_per_sample = self.alpha[targets]  # Shape (batch_size,)
            focal_loss_per_sample = alpha_per_sample * focal_loss_per_sample

        # BƯỚC 6: Gộp kết quả theo reduction
        if self.reduction == "mean":
            return focal_loss_per_sample.mean()
        elif self.reduction == "sum":
            return focal_loss_per_sample.sum()
        else:
            return focal_loss_per_sample  # "none": trả về từng mẫu


class PieceClassifierTrainer:
    """
    Quản lý toàn bộ quá trình huấn luyện mô hình phân loại quân cờ.

    Nhiệm vụ:
        - Khởi tạo loss function, optimizer, scheduler
        - Chạy train / evaluate theo từng epoch
        - Lưu checkpoint khi val_acc tốt hơn
        - Dừng sớm (early stopping) nếu val_acc không cải thiện
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        loss_type: str = "focal_loss",   # "focal_loss" | "weighted_ce" | "cross_entropy"
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs_trained = 0

        # Cài đặt hàm mất mát theo loại yêu cầu
        weights_on_device = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = self._build_loss_function(loss_type, weights_on_device, focal_gamma)

        # AdamW: optimizer với weight decay tích hợp (tốt hơn Adam thông thường)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _build_loss_function(
        self,
        loss_type: str,
        class_weights: Optional[torch.Tensor],
        focal_gamma: float,
    ) -> nn.Module:
        """
        Khởi tạo hàm mất mát theo loại được chọn.

        Lưu ý khi nào dùng loại nào:
          - "focal_loss"   : Khi dữ liệu MẤT CÂN BẰNG NẶNG (vd: 10x chênh lệch)
          - "weighted_ce"  : Khi dữ liệu mất cân bằng nhẹ, muốn training ổn định hơn
          - "cross_entropy": Khi dữ liệu cân bằng tốt (mỗi lớp có số mẫu tương đương)
        """
        if loss_type == "focal_loss":
            return FocalLoss(alpha=class_weights, gamma=focal_gamma)
        elif loss_type == "weighted_ce":
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss()

    def train_epoch(self) -> Tuple[float, float]:
        """
        Chạy 1 epoch huấn luyện đầy đủ.

        Mỗi batch thực hiện:
          1. Forward pass: model(images) -> logits
          2. Tính loss: criterion(logits, labels)
          3. Backward pass: loss.backward() -> tính gradient
          4. Cập nhật trọng số: optimizer.step()

        Returns:
            (loss trung bình, độ chính xác %) trên toàn epoch train
        """
        self.model.train()  # Chuyển model về chế độ train (bật dropout, batch norm)

        total_loss_sum = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Xóa gradient từ batch trước (quan trọng! Nếu quên sẽ cộng dồn gradient)
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(images)

            # Tính loss
            loss = self.criterion(logits, labels)

            # Backward pass và cập nhật trọng số
            loss.backward()
            self.optimizer.step()

            # Cộng dồn để tính trung bình cuối epoch
            # Nhân loss.item() * batch_size vì loss là trung bình của batch
            total_loss_sum += loss.item() * images.size(0)

            # Đếm số mẫu đoán đúng
            predicted_labels = torch.argmax(logits, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

        # Tính loss và accuracy trung bình trên toàn epoch
        epoch_loss = total_loss_sum / max(total_samples, 1)
        epoch_accuracy_percent = (total_correct / max(total_samples, 1)) * 100.0
        return epoch_loss, epoch_accuracy_percent

    @torch.no_grad()  # Tắt gradient vì chỉ evaluate, không update trọng số
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Tuple[float, float]:
        """
        Đánh giá mô hình trên tập validation (không cập nhật trọng số).

        Args:
            data_loader - DataLoader để đánh giá. Nếu None thì dùng self.val_loader.

        Returns:
            (val_loss trung bình, val_accuracy %) trên toàn tập validation
        """
        loader = data_loader or self.val_loader
        if loader is None:
            return 0.0, 0.0

        self.model.eval()  # Chuyển về chế độ eval (tắt dropout, batch norm dùng running stats)

        total_loss_sum = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss_sum += loss.item() * images.size(0)
            predicted_labels = torch.argmax(logits, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

        val_loss = total_loss_sum / max(total_samples, 1)
        val_accuracy_percent = (total_correct / max(total_samples, 1)) * 100.0
        return val_loss, val_accuracy_percent

    def fit(
        self,
        epochs: int = 50,
        save_path: str = "models/stage_c_classifier.pt",
        early_stopping_patience: int = 15,
    ) -> Dict[str, list]:
        """
        Vòng lặp huấn luyện đầy đủ với early stopping và lưu model tốt nhất.

        Cơ chế early stopping:
          Theo dõi val_acc tốt nhất. Nếu N epoch liên tiếp không cải thiện
          (patience_counter >= early_stopping_patience) thì dừng sớm.

        Checkpoint lưu những gì:
          - epoch: số epoch đã train
          - backbone: tên kiến trúc (để PieceClassifier load đúng khi inference)
          - model_state_dict: trọng số model
          - best_val_acc: accuracy tốt nhất (để biết model đang ở level nào)

        Returns:
            Dict history gồm các list: train_loss, train_acc, val_loss, val_acc
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # CosineAnnealingLR: learning rate giảm dần theo hình sin từ lr_max -> 0
        # Tốt hơn StepLR vì không giảm đột ngột, tránh model bị "stuck"
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        best_val_accuracy = 0.0
        patience_counter = 0
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            # Huấn luyện 1 epoch
            train_loss, train_acc = self.train_epoch()

            # Đánh giá trên tập validation
            val_loss, val_acc = self.evaluate()

            # Cập nhật learning rate theo schedule
            lr_scheduler.step()

            # Ghi lại lịch sử để vẽ đồ thị sau
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Kiểm tra xem val_acc có tốt hơn lần trước không
            if val_acc > best_val_accuracy:
                # Model tốt hơn -> lưu checkpoint và reset patience
                best_val_accuracy = val_acc
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "backbone": getattr(self, "backbone_name", "mobilenet_v3_small"),
                        "model_state_dict": self.model.state_dict(),
                        "best_val_acc": best_val_accuracy,
                    },
                    save_path,
                )
                print(f"  -> Đã lưu model tốt nhất vào {save_path} (Val Acc: {best_val_accuracy:.2f}%)")

            else:
                # Không cải thiện -> tăng patience counter
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping: Val Acc không cải thiện sau {early_stopping_patience} epochs. Dừng lại.")
                    break

        return history
