"""
Unit tests kiểm tra các thuật toán hình học: Sắp xếp góc, Khôi phục góc bị che D = A + C - B, Homography và Lưới chuẩn hóa.
"""

import unittest
import numpy as np

from src.stage_a_board.corner_detector import order_corners_clockwise, recover_missing_corner
from src.stage_a_board.canonical_grid import CanonicalBoardGrid
from src.stage_a_board.rectification import HomographyRectifier, detect_board_orientation, rotate_corners_for_landscape


class TestGeometryAndHomography(unittest.TestCase):

    def test_order_corners_clockwise(self):
        """Kiểm tra thứ tự sắp xếp 4 góc theo chiều kim đồng hồ [TL, TR, BR, BL]"""
        raw_pts = np.array([
            [500.0, 500.0],  # BR
            [100.0, 100.0],  # TL
            [100.0, 500.0],  # BL
            [500.0, 100.0],  # TR
        ], dtype=np.float32)

        ordered = order_corners_clockwise(raw_pts)

        np.testing.assert_allclose(ordered[0], [100.0, 100.0])  # TL
        np.testing.assert_allclose(ordered[1], [500.0, 100.0])  # TR
        np.testing.assert_allclose(ordered[2], [500.0, 500.0])  # BR
        np.testing.assert_allclose(ordered[3], [100.0, 500.0])  # BL

    def test_recover_missing_corners_all_four_cases(self):
        """Kiểm tra công thức phục hồi góc bị che cho cả 4 vị trí: TL, TR, BR, BL"""
        # 1 hình bình hành chuẩn: TL(100, 100), TR(500, 120), BR(520, 600), BL(120, 580)
        # Kiểm tra tính chất: TL + BR = (620, 700), TR + BL = (620, 700)
        true_corners = np.array([
            [100.0, 100.0],
            [500.0, 120.0],
            [520.0, 600.0],
            [120.0, 580.0],
        ], dtype=np.float32)

        # Trường hợp 1: Che góc TL (chỉ số 0)
        vis_0 = np.array([0.1, 1.0, 1.0, 1.0])
        rec_0, is_rec_0, idx_0 = recover_missing_corner(true_corners, vis_0, conf_threshold=0.4)
        self.assertTrue(is_rec_0)
        self.assertEqual(idx_0, 0)
        np.testing.assert_allclose(rec_0[0], true_corners[0], atol=1e-4)

        # Trường hợp 2: Che góc TR (chỉ số 1)
        vis_1 = np.array([1.0, 0.0, 1.0, 1.0])
        rec_1, is_rec_1, idx_1 = recover_missing_corner(true_corners, vis_1, conf_threshold=0.4)
        self.assertTrue(is_rec_1)
        self.assertEqual(idx_1, 1)
        np.testing.assert_allclose(rec_1[1], true_corners[1], atol=1e-4)

        # Trường hợp 3: Che góc BR (chỉ số 2)
        vis_2 = np.array([1.0, 1.0, 0.2, 1.0])
        rec_2, is_rec_2, idx_2 = recover_missing_corner(true_corners, vis_2, conf_threshold=0.4)
        self.assertTrue(is_rec_2)
        self.assertEqual(idx_2, 2)
        np.testing.assert_allclose(rec_2[2], true_corners[2], atol=1e-4)

        # Trường hợp 4: Che góc BL (chỉ số 3)
        vis_3 = np.array([1.0, 1.0, 1.0, 0.0])
        rec_3, is_rec_3, idx_3 = recover_missing_corner(true_corners, vis_3, conf_threshold=0.4)
        self.assertTrue(is_rec_3)
        self.assertEqual(idx_3, 3)
        np.testing.assert_allclose(rec_3[3], true_corners[3], atol=1e-4)

    def test_canonical_grid_generation(self):
        """Kiểm tra việc tạo 90 giao điểm chuẩn hóa"""
        grid = CanonicalBoardGrid(canvas_width=720, canvas_height=800, margin_x=40, margin_y=40)
        self.assertEqual(grid.grid_points_2d.shape, (10, 9, 2))
        self.assertEqual(grid.grid_points_flat.shape, (90, 2))

        # Kiểm tra tọa độ 4 góc ngoài cùng của lưới
        tl_x, tl_y = grid.get_pixel_coord(0, 0)
        self.assertAlmostEqual(tl_x, 40.0)
        self.assertAlmostEqual(tl_y, 40.0)

        br_x, br_y = grid.get_pixel_coord(8, 9)
        self.assertAlmostEqual(br_x, 720.0 - 40.0)
        self.assertAlmostEqual(br_y, 800.0 - 40.0)

    def test_homography_forward_and_inverse(self):
        """Kiểm tra tính nhất quán của phép biến đổi Homography 2 chiều"""
        grid = CanonicalBoardGrid()
        rectifier = HomographyRectifier(grid)

        src_corners = np.array([
            [120.0, 80.0],
            [900.0, 90.0],
            [960.0, 700.0],
            [60.0, 680.0],
        ], dtype=np.float32)

        H = rectifier.compute_homography(src_corners)
        self.assertEqual(H.shape, (3, 3))

        # Điểm gốc -> Canonical -> Gốc
        test_pts = np.array([[500.0, 400.0], [200.0, 150.0]], dtype=np.float32)
        warped_pts = rectifier.transform_points(test_pts)
        restored_pts = rectifier.inverse_transform_points(warped_pts)

        np.testing.assert_allclose(restored_pts, test_pts, atol=1e-3)


    def test_detect_board_orientation_portrait(self):
        """Bàn cờ dọc: chiều cao > chiều rộng -> orientation = 'portrait'"""
        # Hình chữ nhật đứng: rộng 400px, cao 600px
        portrait_corners = np.array([
            [100.0, 100.0],  # TL
            [500.0, 100.0],  # TR
            [500.0, 700.0],  # BR
            [100.0, 700.0],  # BL
        ], dtype=np.float32)

        orientation = detect_board_orientation(portrait_corners)
        self.assertEqual(orientation, "portrait")

    def test_detect_board_orientation_landscape(self):
        """Bàn cờ ngang: chiều rộng > chiều cao -> orientation = 'landscape'"""
        # Hình chữ nhật nằm ngang: rộng 600px, cao 400px
        landscape_corners = np.array([
            [100.0, 200.0],  # TL
            [700.0, 200.0],  # TR
            [700.0, 600.0],  # BR
            [100.0, 600.0],  # BL
        ], dtype=np.float32)

        orientation = detect_board_orientation(landscape_corners)
        self.assertEqual(orientation, "landscape")

    def test_rotate_corners_for_landscape(self):
        """
        Kiểm tra xoay góc: Sau khi xoay bàn cờ ngang, góc TL mới phải là BL cũ.
        Trực quan: BL_cũ -> TL_mới, TL_cũ -> TR_mới, TR_cũ -> BR_mới, BR_cũ -> BL_mới
        """
        landscape_corners = np.array([
            [100.0, 200.0],  # TL
            [700.0, 200.0],  # TR
            [700.0, 600.0],  # BR
            [100.0, 600.0],  # BL
        ], dtype=np.float32)

        rotated = rotate_corners_for_landscape(landscape_corners)

        # TL mới = BL cũ
        np.testing.assert_allclose(rotated[0], [100.0, 600.0])  # new TL = old BL
        # TR mới = TL cũ
        np.testing.assert_allclose(rotated[1], [100.0, 200.0])  # new TR = old TL
        # BR mới = TR cũ
        np.testing.assert_allclose(rotated[2], [700.0, 200.0])  # new BR = old TR
        # BL mới = BR cũ
        np.testing.assert_allclose(rotated[3], [700.0, 600.0])  # new BL = old BR

    def test_homography_landscape_produces_portrait_output(self):
        """
        Kiểm tra end-to-end: Bàn cờ nằm ngang khi qua Homography phải cho ra ảnh dọc đúng tỉ lệ.
        Cụ thể: Điểm TL của canonical phải nằm ở góc trên-trái canvas.
        """
        grid = CanonicalBoardGrid(canvas_width=720, canvas_height=800, margin_x=40, margin_y=40)
        rectifier = HomographyRectifier(grid)

        # Bàn cờ nằm ngang (rộng 800, cao 500 trong ảnh gốc)
        landscape_src = np.array([
            [50.0,  100.0],   # TL
            [850.0, 100.0],   # TR
            [850.0, 600.0],   # BR
            [50.0,  600.0],   # BL
        ], dtype=np.float32)

        H = rectifier.compute_homography(landscape_src)

        # Orientation phải được phát hiện là landscape
        self.assertEqual(rectifier.detected_orientation, "landscape")

        # Ma trận H phải hợp lệ (3x3, không singular)
        self.assertEqual(H.shape, (3, 3))
        self.assertGreater(abs(np.linalg.det(H)), 1e-6)


if __name__ == "__main__":
    unittest.main()

