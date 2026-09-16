# Hệ Thống Nhận Diện Bàn Cờ Tướng (China Chess Computer Vision)

Một kiến trúc thị giác máy tính phân rã theo cấu trúc hình học bàn cờ tướng ($9 \times 10$), loại bỏ hiện tượng "học vẹt" của mô hình End-to-End và cho phép kiểm soát, debug độc lập từng loại sai số.

---

## 1. Vấn Đề Gốc & Triết Lý Thiết Kế

### Nhược điểm của việc ném thẳng ảnh vào YOLO thông thường:
- Khi một mô hình duy nhất phải vừa tìm vị trí, vừa đọc chữ trên quân cờ, vừa thích nghi với mọi góc chụp nghiêng và ánh sáng:
  - **Học vẹt**: Model ghi nhớ các đặc điểm nền bàn cờ, góc đặt camera hoặc bộ cờ cụ thể của tập train.
  - **Không thể debug**: Khi nhận diện sai, không thể biết do méo phối cảnh, do phát hiện thiếu bbox, hay do chữ trên quân cờ khó đọc.
  - **Mất cân bằng dữ liệu tự nhiên**: Tướng chỉ có 1 quân trong khi Tốt có 5 quân, khiến detector bị thiên lệch.

### Giải pháp: Phân rã bài toán thành 3 giai đoạn độc lập (A -> B -> C) + Hậu xử lý hình học

```text
+-----------------------------------------------------------------------------------------+
|                                    LUỒNG XỬ LÝ CHÍNH                                    |
|                                                                                         |
|   Ảnh camera / Video Stream                                                             |
|              │                                                                          |
|              ▼                                                                          |
|   ┌───────────────────────┐                                                             |
|   │    Change Detector    │ ───► Bỏ qua frame khi tay người đang di chuyển              |
|   │  & Debounce (Ổn định) │ ───► Kích hoạt nhận diện khi bàn cờ tĩnh >= 0.5s            |
|   └──────────┬────────────┘                                                             |
|              │                                                                          |
|              ▼                                                                          |
|   ┌────────────────────────────────────────────────────────┐                            |
|   │ Giai đoạn A: Định vị 4 góc & Khử méo phối cảnh         │                            |
|   │ - YOLO-Pose detect 4 góc [TL, TR, BR, BL]              │                            |
|   │ - Khôi phục góc bị che: D = A + C - B                  │                            |
|   │ - Camera Undistort + Homography Warp về Canvas 720x800 │                            |
|   └──────────────────────────┬─────────────────────────────┘                            |
|                              │ (Ảnh Top-Down Chuẩn Hóa)                                 |
|                              ▼                                                          |
|   ┌────────────────────────────────────────────────────────┐                            |
|   │ Giai đoạn B: Định vị quân cờ (Piece Detection)         │                            |
|   │ - YOLO-nano chỉ tìm vị trí (BBox đồng nhất kích thước) │                            |
|   └──────────────────────────┬─────────────────────────────┘                            |
|                              │ (Crop các Patch 64x64)                                   |
|                              ▼                                                          |
|   ┌────────────────────────────────────────────────────────┐                            |
|   │ Giai đoạn C: Phân loại 14 loại quân cờ                 │                            |
|   │ - MobileNetV3 / Custom CNN phân loại chi tiết chữ Hán  │                            |
|   │ - Xử lý mất cân bằng lớp bằng Focal Loss / Class Weight│                            |
|   └──────────────────────────┬─────────────────────────────┘                            |
|                              │                                                          |
|                              ▼                                                          |
|   ┌────────────────────────────────────────────────────────┐                            |
|   │ Hậu xử lý (Rule-based Post-processing)                 │                            |
|   │ - KD-Tree Snap tâm quân vào 90 giao điểm lưới 9x10     │                            |
|   │ - Áp constraint số lượng tối đa (Tướng: 1, Sĩ/Tượng: 2)│                            |
|   │ - Xuất chuỗi chuẩn Xiangqi FEN & Ma trận trạng thái    │                            |
|   └────────────────────────────────────────────────────────┘                            |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
China_chess_cv/
├── configs/
│   ├── stage_a_board.yaml          # Cấu hình định vị góc bàn & Homography
│   ├── stage_b_pieces.yaml         # Cấu hình YOLO detect quân cờ
│   ├── stage_c_classifier.yaml     # Cấu hình phân loại 14 lớp quân & Focal Loss
│   └── pipeline_config.yaml        # Cấu hình toàn diện, Change Detection & Debounce
├── data/
│   └── synthetic/                  # Dữ liệu tổng hợp (A, B, C)
├── models/                         # Trọng số mô hình (.pt / .onnx)
├── src/
│   ├── stage_a_board/              # [Giai đoạn A] Định vị bàn cờ & Homography
│   │   ├── canonical_grid.py       # Tọa độ 90 giao điểm chuẩn trên canvas 720x800
│   │   ├── rectification.py        # Undistort ống kính & Warp Homography
│   │   └── corner_detector.py      # YOLO-Pose & Thuật toán phục hồi góc D = A + C - B
│   ├── stage_b_piece_detect/       # [Giai đoạn B] Định vị quân cờ
│   │   └── piece_detector.py       # BBox detector & trích xuất patch vuông
│   ├── stage_c_piece_classify/     # [Giai đoạn C] Phân loại 14 lớp quân cờ
│   │   ├── model.py                # MobileNetV3-Small & CustomLightweightCNN
│   │   ├── dataset.py              # Dataset, Augmentation xoay 360 độ & Class Weights
│   │   ├── trainer.py              # Training loop với Focal Loss & Cosine Scheduler
│   │   └── classifier.py           # Inference wrapper
│   ├── post_processing/            # Hậu xử lý & Ràng buộc luật cờ
│   │   ├── piece_definitions.py    # Danh mục 14 quân cờ, ký hiệu FEN, chữ Hán
│   │   ├── grid_snapper.py         # KD-Tree snap vào lưới 9x10 & giải quyết xung đột
│   │   └── board_state.py          # Quản lý ma trận 10x9 & xuất chuỗi Xiangqi FEN
│   ├── realtime/                   # Tối ưu hóa thời gian thực
│   │   ├── change_detector.py      # Phát hiện chuyển động & Bộ đếm debounce ổn định
│   │   └── stream_pipeline.py      # Điều phối luồng xử lý camera thời gian thực
│   ├── synthetic/                  # Bộ sinh dữ liệu tổng hợp
│   │   ├── board_renderer.py       # Sinh ảnh bàn cờ 2D/3D kèm vân gỗ, chữ Hán, góc nghiêng
│   │   └── dataset_generator.py    # Xuất dataset đồng bộ cho Stage A, B, C
│   ├── evaluation/                 # Đo lường & Benchmark từng giai đoạn độc lập
│   │   ├── metrics_stage_a.py      # Sai số góc L2 (MAE/RMSE) & Reprojection Error
│   │   ├── metrics_stage_b.py      # Precision, Recall, F1, Sót quân, Quân ma
│   │   ├── metrics_stage_c.py      # Top-1 Accuracy, Confusion Matrix 14x14
│   │   └── evaluate_all.py         # Đánh giá toàn diện End-to-End
│   └── pipeline.py                 # Pipeline End-to-End hoàn chỉnh
├── scripts/
│   ├── generate_synthetic_data.py  # CLI sinh dataset tổng hợp
│   ├── train_stage_c.py            # CLI huấn luyện Stage C
│   ├── run_inference.py            # CLI chạy nhận diện ảnh đơn / Webcam realtime
│   └── benchmark.py                # CLI kiểm thử & xuất báo cáo đánh giá
├── tests/                          # Bộ Unit Tests tự động
├── requirements.txt                # Danh sách thư viện phụ thuộc
└── README.md
```

---

## 3. Danh Mục 14 Loại Quân Cờ Chuẩn

| Class ID | Tên Mã (`name_code`) | FEN Char | Chữ Hán (Phồn thể / Giản thể) | Tên Tiếng Việt | Số Lượng Tối Đa |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | `r_king` | `K` | 帥 / 帅 | Tướng Đỏ | 1 |
| **1** | `r_advisor` | `A` | 仕 | Sĩ Đỏ | 2 |
| **2** | `r_elephant` | `B` | 相 | Tượng Đỏ | 2 |
| **3** | `r_horse` | `N` | 傌 / 马 | Mã Đỏ | 2 |
| **4** | `r_chariot` | `R` | 俥 / 车 | Xe Đỏ | 2 |
| **5** | `r_cannon` | `C` | 炮 | Pháo Đỏ | 2 |
| **6** | `r_soldier` | `P` | 兵 | Binh/Tốt Đỏ | 5 |
| **7** | `b_king` | `k` | 將 / 将 | Tướng Đen | 1 |
| **8** | `b_advisor` | `a` | 士 | Sĩ Đen | 2 |
| **9** | `b_elephant` | `b` | 象 | Tượng Đen | 2 |
| **10** | `b_horse` | `n` | 馬 / 马 | Mã Đen | 2 |
| **11** | `b_chariot` | `r` | 車 / 车 | Xe Đen | 2 |
| **12** | `b_cannon` | `c` | 砲 / 炮 | Pháo Đen | 2 |
| **13** | `b_soldier` | `p` | 卒 | Tốt Đen | 5 |

---

## 4. Hướng Dẫn Cài Đặt & Sử Dụng

### 4.1 Cài đặt môi trường
```bash
py -3.11 -m pip install -r requirements.txt
```

### 4.2 Sinh dữ liệu tổng hợp (Synthetic Pretraining Data)
Sinh tự động hàng loạt ảnh bàn cờ kèm góc chụp nghiêng 3D, ánh sáng, nhiễu và nhãn ground-truth đồng bộ cho cả 3 giai đoạn:
```bash
py -3.11 scripts/generate_synthetic_data.py --num_train 200 --num_val 40 --output_dir data/synthetic --occlusion_prob 0.25
```

### 4.3 Huấn luyện bộ phân loại 14 lớp quân cờ (Stage C)
```bash
py -3.11 scripts/train_stage_c.py --data_dir data/synthetic/stage_c_classifier --epochs 30 --batch_size 64 --loss focal_loss --backbone mobilenet_v3_small
```

### 4.4 Chạy nhận diện (Inference)

#### Nhận diện trên ảnh tĩnh đơn lẻ:
```bash
py -3.11 scripts/run_inference.py --image data/synthetic/stage_a_board/images/val/val_00000.jpg --output_vis output_result.jpg
```

#### Chạy luồng Webcam thời gian thực với Change Detection & Debounce:
```bash
py -3.11 scripts/run_inference.py --webcam 0
```
> **Cơ chế**: Giai đoạn A chỉ chạy định vị lúc setup ban đầu. Giai đoạn B và C chỉ kích hoạt khi người chơi đã thực hiện xong nước đi và rút tay khỏi bàn cờ (bàn cờ ổn định liên tiếp $\ge 15$ frames), giúp tiết kiệm tài nguyên tính toán và loại bỏ nhiễu lúc tay đang chuyển động.

### 4.5 Kiểm thử & Benchmark từng giai đoạn độc lập
```bash
py -3.11 scripts/benchmark.py --num_test_boards 30
```
Báo cáo sẽ hiển thị độc lập:
- **Giai đoạn A**: Sai số định vị góc bàn (MAE/RMSE pixel error).
- **Giai đoạn B**: Precision, Recall, F1, Tỷ lệ bỏ sót quân (Miss Rate), Tỷ lệ quân ma (Ghost Rate).
- **Giai đoạn C**: Top-1 Accuracy, Confusion Matrix, Tỷ lệ nhầm màu (Red vs Black).
- **End-to-End**: Độ chính xác 90 giao điểm và Tỷ lệ ván cờ khớp FEN tuyệt đối 100%.

### 4.6 Chạy Unit Tests
```bash
py -3.11 -m unittest discover tests
```

---

## 5. Bảng Công Thức Toán Học & Thuật Toán Cốt Lõi (Định dạng thuần dễ đọc)

```text
================================================================================
1. PHỤC HỒI GÓC BÀN BỊ CHE (D = A + C - B):
   TL + BR = TR + BL
   - Nếu mất góc Top-Left (TL):     TL = TR + BL - BR
   - Nếu mất góc Top-Right (TR):    TR = TL + BR - BL
   - Nếu mất góc Bottom-Right (BR): BR = TR + BL - TL
   - Nếu mất góc Bottom-Left (BL):  BL = TL + BR - TR

================================================================================
2. KHỬ MÉO ỐNG KÍNH CAMERA (Brown-Conrady Model):
   Với r^2 = x^2 + y^2:
   x_undist = x * (1 + k1*r^2 + k2*r^4 + k3*r^6) + 2*p1*x*y + p2*(r^2 + 2*x^2)
   y_undist = y * (1 + k1*r^2 + k2*r^4 + k3*r^6) + p1*(r^2 + 2*y^2) + 2*p2*x*y

================================================================================
3. BIẾN ĐỔI PHỐI CẢNH (Homography Warp sang Canvas chuẩn 720x800):
   [x_canvas, y_canvas, 1]^T = H * [x_camera, y_camera, 1]^T

================================================================================
4. TỌA ĐỘ 90 GIAO ĐIỂM CHUẨN (Lưới 9 cột x 10 hàng trên Canvas 720x800):
   - Bước ngang: step_x = (720 - 2 * 40) / (9 - 1)  = 640 / 8 = 80 pixel
   - Bước dọc:   step_y = (800 - 2 * 40) / (10 - 1) = 720 / 9 = 80 pixel
   - Tọa độ tâm giao điểm (cột c, hàng r):
     x(c) = 40 + c * 80   (với c từ 0 đến 8)
     y(r) = 40 + r * 80   (với r từ 0 đến 9)

================================================================================
5. SNAP GIAO ĐIỂM GẦN NHẤT (Grid Snapping):
   Khoảng cách d = sqrt( (x_quan - x(c))^2 + (y_quan - y(r))^2 )
   - Chọn giao điểm (c, r) có khoảng cách d nhỏ nhất.
   - Điều kiện hợp lệ: d <= 38 pixel (nếu d > 38px: loại bỏ vì ngoài bàn cờ).

================================================================================
6. FOCAL LOSS (Phân loại 14 lớp, giải quyết mẫu khó):
   FL(p_t) = - alpha_t * ((1 - p_t) ^ gamma) * ln(p_t)
   - p_t:     Xác suất mô hình đoán đúng nhãn thực tế (từ 0.0 đến 1.0)
   - gamma:   Hệ số điều chế (chọn gamma = 2.0)
   - alpha_t: Trọng số cân bằng lớp (lấy từ w_c)
   => Khi mẫu dễ (p_t = 0.95): (1 - 0.95)^2 = 0.0025 (Giảm Loss 400 lần)
   => Khi mẫu khó (p_t = 0.20): (1 - 0.20)^2 = 0.6400 (Giữ lại 64% Loss)

================================================================================
7. TRỌNG SỐ NGHỊCH ĐẢO CÂN BẰNG LỚP (Inverse Class Frequency):
   w_c = Tong_so_mau / (14 * So_mau_lop_c)
   - Tướng (1 quân):  w_king    = 32 / (14 * 1) = 2.2857 (Tăng trọng số gấp 5 lần)
   - Sĩ/Tượng/Mã/Xe/Pháo (2):   = 32 / (14 * 2) = 1.1428
   - Tốt/Binh (5 quân): w_pawn  = 32 / (14 * 5) = 0.4571 (Giảm trọng số)

================================================================================
8. PHÁT HIỆN CHUYỂN ĐỘNG & DEBOUNCE (Change Detection):
   Motion_Score (%) = (So_pixel_bien_doi / Tong_so_pixel) * 100
   - Nếu Motion_Score >= 15%: Đang di chuyển tay -> Tạm dừng nhận diện.
   - Nếu Motion_Score < 15% liên tiếp >= 15 frames (~0.5s): Bàn cờ tĩnh -> Chạy B+C.
================================================================================
```
