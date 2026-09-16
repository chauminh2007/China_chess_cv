"""
Đánh giá độc lập Giai đoạn A: Độ chính xác định vị 4 góc và ma trận Homography.
"""

from typing import List, Dict, Tuple
import numpy as np
import cv2


def compute_corner_l2_errors(
    pred_corners: np.ndarray,
    gt_corners: np.ndarray,
) -> Dict[str, float]:
    """
    Tính sai số khoảng cách L2 (pixel) giữa 4 góc dự đoán và 4 góc ground-truth.
    
    Args:
        pred_corners: shape (4, 2) hoặc (N, 4, 2)
        gt_corners: shape (4, 2) hoặc (N, 4, 2)
        
    Returns:
        Dictionary chứa MAE, RMSE, Max Error
    """
    p = np.array(pred_corners, dtype=np.float32).reshape(-1, 4, 2)
    g = np.array(gt_corners, dtype=np.float32).reshape(-1, 4, 2)

    # Khoảng cách L2 cho từng góc
    dists = np.linalg.norm(p - g, axis=-1)  # shape (N, 4)

    mean_l2 = float(np.mean(dists))
    rmse_l2 = float(np.sqrt(np.mean(dists ** 2)))
    max_l2 = float(np.max(dists))

    # Sai số theo từng góc riêng biệt (TL, TR, BR, BL)
    per_corner_mean = np.mean(dists, axis=0)

    return {
        "mean_corner_error_px": mean_l2,
        "rmse_corner_error_px": rmse_l2,
        "max_corner_error_px": max_l2,
        "tl_error_px": float(per_corner_mean[0]),
        "tr_error_px": float(per_corner_mean[1]),
        "br_error_px": float(per_corner_mean[2]),
        "bl_error_px": float(per_corner_mean[3]),
    }


def compute_homography_reprojection_error(
    pred_H: np.ndarray,
    gt_H: np.ndarray,
    canonical_grid_points: np.ndarray,
) -> float:
    """
    Tính sai số chiếu lại (Reprojection Error) trung bình trên toàn bộ 90 giao điểm bàn cờ.
    """
    pts = np.array(canonical_grid_points, dtype=np.float32).reshape(-1, 1, 2)

    # Chiếu ngược về ảnh phối cảnh bằng H_inv
    inv_pred = np.linalg.inv(pred_H)
    inv_gt = np.linalg.inv(gt_H)

    proj_pred = cv2.perspectiveTransform(pts, inv_pred).reshape(-1, 2)
    proj_gt = cv2.perspectiveTransform(pts, inv_gt).reshape(-1, 2)

    error = np.mean(np.linalg.norm(proj_pred - proj_gt, axis=1))
    return float(error)
