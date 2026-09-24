"""
camera_capture.py
─────────────────
Tiện ích chụp ảnh từ camera, dùng để thu thập dữ liệu thật upload lên Roboflow label.

Quy trình Roboflow:
    1. Chụp ảnh  : python camera_capture.py --prefix jovi
                     → ảnh lưu vào  data/raw/images/
    2. Upload     : kéo data/raw/images/ vào Roboflow project tương ứng
    3. Label      : label trên Roboflow UI
    4. Export     : xuất dataset từ Roboflow, giải nén vào đúng thư mục dưới đây:

    │  Mục đích          │  Thư mục đích sau Roboflow export
    ├───────────────────┼───────────────────────────────────────────────────
    │  Stage A (góc bàn) │  data/stage_a/{train,val}/{images,labels}/
    │  Stage B (quân cờ)  │  data/stage_b/{train,val}/{images,labels}/
    │  Stage C (phân loại)│  data/stage_c/{train,val}/<tên_lớp>/*.jpg

Cách dùng:
    python camera_capture.py                          # tự động quét camera
    python camera_capture.py --camera 1 --prefix jovi # cam jovi, tên file jovi_...
    python camera_capture.py --list                   # liệt kê camera rồi thoát

Phím tắt:
    SPACE / s   → Chụp ảnh
    q / ESC     → Thoát
    r           → Xoay ảnh 90°
    f           → Lật ảnh ngang (horizontal flip)
    v           → Lật ảnh dọc (vertical flip)
    +/-         → Tăng/giảm chất lượng JPEG
"""

import argparse
import os
import sys
import time
import subprocess
from typing import Union
from datetime import datetime
from pathlib import Path

import cv2


def get_windows_camera_names() -> list[str]:
    """Sử dụng PowerShell để lấy danh sách tên các camera trên Windows."""
    names = []
    if os.name == 'nt':
        try:
            # Tìm trong các class chứa camera local
            classes = ["Camera", "Image"]
            for cls in classes:
                cmd = f'powershell -Command "Get-PnpDevice -Class {cls} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FriendlyName"'
                try:
                    out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
                    if out:
                        for line in out.split('\n'):
                            line = line.strip()
                            if line and line not in names:
                                names.append(line)
                except Exception:
                    pass
        except Exception:
            pass
    return names


def detect_cameras(max_index: int = 5) -> list[dict]:
    """Quét camera khả dụng. Sử dụng CAP_ANY để đảm bảo chọn đúng index."""
    cameras = []
    print("Đang quét camera", end="", flush=True)
    
    # Bỏ DSHOW vì nó gây lỗi bỏ qua index trên Windows
    backend = cv2.CAP_ANY
    
    # Lấy tên hiển thị của camera (friendly name)
    cam_names = get_windows_camera_names()
    
    opened_count = 0
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Gán tên theo thứ tự mở thành công để tránh lỗi map sai index của Windows
            friendly_name = cam_names[opened_count] if opened_count < len(cam_names) else f"Camera {idx}"
            opened_count += 1
            
            cameras.append({
                "index": idx,
                "name": friendly_name,
                "width": w,
                "height": h,
                "fps": fps,
                "backend": cap.getBackendName(),
            })
        cap.release()
        print(".", end="", flush=True)
    print()
    return cameras


def print_camera_list(cameras: list[dict]) -> None:
    if not cameras:
        print("  [!] Không tìm thấy camera nào.")
        return
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                          DANH SÁCH CAMERA KHẢ DỤNG                          │")
    print("├───────┬──────────────────────────────┬───────────────┬───────────┬──────────┤")
    print("│ Index │ Tên Thiết Bị                 │  Độ phân giải │    FPS    │ Backend  │")
    print("├───────┼──────────────────────────────┼───────────────┼───────────┼──────────┤")
    for cam in cameras:
        fps_str = f"{cam['fps']:.1f}" if cam["fps"] > 0 else "N/A"
        res_str = f"{cam['width']}x{cam['height']}"
        name = cam['name'][:28] # Cắt bớt nếu tên quá dài
        print(f"│  [{cam['index']}]  │ {name:<28} │ {res_str:>13} │ {fps_str:>9} │ {cam['backend']:>8} │")
    print("└───────┴──────────────────────────────┴───────────────┴───────────┴──────────┘\n")


def draw_overlay(frame, cam_display: str, capture_count: int, quality: int,
                 rotation: int, h_flipped: bool, v_flipped: bool, flash_remaining: float):
    h, w = frame.shape[:2]

    # Thanh trạng thái
    cv2.rectangle(frame, (0, 0), (w, 34), (20, 20, 20), -1)
    status = f"{cam_display} | Anh: {capture_count} | Quality: {quality}% | Xoay: {rotation} | H-Lat: {'On' if h_flipped else 'Off'} | V-Lat: {'On' if v_flipped else 'Off'}"
    cv2.putText(frame, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # Hiệu ứng flash
    if flash_remaining > 0:
        alpha = min(1.0, flash_remaining / 0.15)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, alpha * 0.5, frame, 1 - alpha * 0.5, 0, frame)
        cv2.putText(frame, "Chup anh thanh cong!", (w // 2 - 120, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


def run_capture_session(cam_source: Union[int, str], output_dir: Path, quality: int = 90,
                        file_prefix: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(cam_source, str) and (cam_source.startswith("http") or cam_source.startswith("rtsp")):
        backend = cv2.CAP_ANY
        cam_display = "IP_Cam"
        # Dùng file_prefix nếu có, ngược lại dùng 'cam_ip'
        file_prefix = file_prefix or "cam_ip"
    else:
        # Sử dụng DSHOW trên Windows để mở camera local nhanh hơn
        # Phải cast sang int Ở ĐÂY (sau khi đã xác định không phải IP URL),
        # không cast sớm bên ngoài tránh crash khi cam_source là chuỗi URL.
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        cam_source = int(cam_source)
        cam_display = f"CAM[{cam_source}]"
        file_prefix = file_prefix or f"cam{cam_source}"

    cap = cv2.VideoCapture(cam_source, backend)
    
    if not cap.isOpened():
        print(f"[LỖI] Không thể kết nối tới camera [{cam_source}].")
        return

    capture_count = 0
    rotation = 0
    h_flipped = False
    v_flipped = False
    flash_end = 0.0

    print(f"\n  [✓] Đã kết nối camera [{cam_source}].")
    print("  ┌─────────────────────────────────────────┐")
    print("  │       PHÍM TẮT TRONG CỬA SỔ CAMERA      │")
    print("  ├─────────────────────────────────────────┤")
    print("  │ SPACE / S   : Chụp ảnh                  │")
    print("  │ R           : Xoay ảnh 90 độ            │")
    print("  │ F           : Lật ảnh theo chiều ngang  │")
    print("  │ V           : Lật ảnh theo chiều dọc    │")
    print("  │ + / -       : Tăng/giảm chất lượng ảnh  │")
    print("  │ Q / ESC     : Thoát                     │")
    print("  └─────────────────────────────────────────┘\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if h_flipped and v_flipped:
            frame = cv2.flip(frame, -1)
        elif h_flipped:
            frame = cv2.flip(frame, 1)
        elif v_flipped:
            frame = cv2.flip(frame, 0)
            
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        display = frame.copy()
        flash_remaining = max(0.0, flash_end - time.time())
        draw_overlay(display, cam_display, capture_count, quality, rotation, h_flipped, v_flipped, flash_remaining)

        cv2.imshow("Camera Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord(" "), ord("s")):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = output_dir / f"{file_prefix}_{timestamp}.jpg"
            success = cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if success:
                capture_count += 1
                flash_end = time.time() + 0.3
                print(f"  [✓] Lưu: {filename}")
        elif key == ord("r"):
            rotation = (rotation + 90) % 360
        elif key == ord("f"):
            h_flipped = not h_flipped
        elif key == ord("v"):
            v_flipped = not v_flipped
        elif key in (ord("+"), ord("=")):
            quality = min(100, quality + 5)
        elif key == ord("-"):
            quality = max(10, quality - 5)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Tổng số ảnh: {capture_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chụp ảnh từ camera nhanh chóng.")
    parser.add_argument("--camera", "-c", type=str, default=None, help="Index camera (VD: 0) hoặc IP URL (VD: http://192.168.1.5:4747/video).")
    parser.add_argument("--output", "-o", type=str, default="data/raw/images",
                        help="Thư mục lưu ảnh (mặc định: data/raw/images — upload thư mục này lên Roboflow).")
    parser.add_argument("--quality", "-q", type=int, default=90, help="Chất lượng JPEG (1-100).")
    parser.add_argument("--prefix", "-p", type=str, default="", help="Tiền tố tên file ảnh (VD: jovi → jovi_20240924_....jpg). Mặc định: cam<index>.")
    parser.add_argument("--list", "-l", action="store_true", help="Chỉ liệt kê camera.")
    args = parser.parse_args()

    # Nếu truyền --camera, bỏ qua bước quét
    if args.camera is not None:
        run_capture_session(args.camera, Path(args.output), args.quality, args.prefix)
        return

    # Quét tối đa 5 camera
    cameras = detect_cameras(max_index=5)
    if args.list:
        print_camera_list(cameras)
        return

    if not cameras:
        print("\n[!] Không tìm thấy camera local nào được cắm vào máy.")
    else:
        print_camera_list(cameras)

    valid_indices = [str(c["index"]) for c in cameras]

    while True:
        prompt_text = f"  Nhập index camera {valid_indices} hoặc nhập thẳng IP Stream (VD: http://192.168.../video): "
        if len(cameras) == 1:
            prompt_text = f"  Nhấn ENTER để dùng camera [{valid_indices[0]}], hoặc nhập thẳng IP Stream (http://...): "
            
        choice = input(prompt_text).strip()
        
        if not choice and len(cameras) == 1:
            cam_source = int(valid_indices[0])
            break
        elif choice in valid_indices:
            cam_source = int(choice)
            break
        elif choice.startswith("http") or choice.startswith("rtsp"):
            cam_source = choice
            break
        elif choice:
            print("  [!] Lựa chọn không hợp lệ. Vui lòng nhập số index đúng hoặc một đường dẫn http://...")
            
    run_capture_session(cam_source, Path(args.output), args.quality, args.prefix)


if __name__ == "__main__":
    main()
