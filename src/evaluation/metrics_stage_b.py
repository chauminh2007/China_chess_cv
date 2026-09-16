"""
Đánh giá độc lập Giai đoạn B: Độ chính xác định vị vị trí quân cờ (Precision, Recall, Miss Rate, Ghost Rate).
"""

from typing import List, Dict, Tuple
import numpy as np
from scipy.spatial.distance import cdist


def compute_piece_detection_metrics(
    pred_centers: List[Tuple[float, float]],
    gt_centers: List[Tuple[float, float]],
    distance_threshold_px: float = 25.0,
) -> Dict[str, float]:
    """
    Tính Precision, Recall, F1, Tỷ lệ bỏ sót quân (Miss Rate), Tỷ lệ nhận diện ảo (Ghost Rate).
    
    Args:
        pred_centers: Danh sách tọa độ tâm quân cờ dự đoán (x, y)
        gt_centers: Danh sách tọa độ tâm quân cờ thực tế (x, y)
        distance_threshold_px: Ngưỡng khoảng cách tối đa để tính là True Positive
        
    Returns:
        Dictionary chứa các chỉ số đo lường
    """
    num_pred = len(pred_centers)
    num_gt = len(gt_centers)

    if num_gt == 0 and num_pred == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "miss_rate": 0.0, "ghost_rate": 0.0}
    if num_gt == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "miss_rate": 0.0, "ghost_rate": 1.0}
    if num_pred == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "miss_rate": 1.0, "ghost_rate": 0.0}

    preds_arr = np.array(pred_centers, dtype=np.float32)
    gts_arr = np.array(gt_centers, dtype=np.float32)

    # Tính ma trận khoảng cách giữa tất cả các cặp (N_pred, N_gt)
    dist_mat = cdist(preds_arr, gts_arr)

    # Khớp cặp Greedy theo khoảng cách nhỏ nhất
    matched_gt = set()
    matched_pred = set()

    # Sắp xếp các cặp theo khoảng cách tăng dần
    indices = np.dstack(np.unravel_index(np.argsort(dist_mat.ravel()), dist_mat.shape))[0]

    localization_errors = []

    for p_idx, g_idx in indices:
        if p_idx in matched_pred or g_idx in matched_gt:
            continue
        d = dist_mat[p_idx, g_idx]
        if d <= distance_threshold_px:
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)
            localization_errors.append(d)

    tp = len(matched_gt)
    fp = num_pred - tp  # Quân ma (Ghost detections)
    fn = num_gt - tp    # Quân bị sót (Missed pieces)

    precision = tp / max(num_pred, 1)
    recall = tp / max(num_gt, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
    miss_rate = fn / max(num_gt, 1)
    ghost_rate = fp / max(num_pred, 1) if num_pred > 0 else 0.0
    mean_loc_error = float(np.mean(localization_errors)) if localization_errors else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "miss_rate": float(miss_rate),
        "ghost_rate": float(ghost_rate),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "mean_loc_error_px": mean_loc_error,
    }
