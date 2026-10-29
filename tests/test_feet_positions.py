"""Check production coordinate projection without loading learned detector models."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import cv2
    from scipy import signal
except ImportError:
    cv2 = signal = None


def production_method(file, class_name, method_name, scope):
    source = Path(__file__).resolve().parents[1] / file
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    return scope[method_name]


@unittest.skipUnless(cv2 is not None and signal is not None, 'requires OpenCV and SciPy')
class FeetPositionTests(unittest.TestCase):
    def setUp(self):
        self.feet = production_method('detection.py', 'DetectionModel', 'calculate_feet_positions',
                                      {'np': np, 'cv2': cv2, 'signal': signal})
        self.heatmap = production_method('statistics_module.py', 'Statistics', 'get_player_position_heatmap',
                                         {'np': np})

    def positions(self, missing=(), top_box=(80, 10, 100, 30), inverse=None):
        bottom = [np.array([20, 20, 40, 60], dtype=float) for _ in range(9)]
        top = [[None] * 4 if i in missing else np.array(top_box, dtype=float) for i in range(9)]
        model = SimpleNamespace(player_1_boxes=bottom, player_2_boxes=top)
        court = SimpleNamespace(game_warp_matrix=[np.eye(3) if inverse is None else inverse] * 9,
                                court_reference=SimpleNamespace(court_total_width=160, court_total_height=160))
        bottom, top = self.feet(model, court)
        stats = SimpleNamespace(court_tracker=court, feet_bottom=bottom, feet_top=top)
        return bottom, top, self.heatmap(stats)

    def test_observed_positions_keep_projection_and_frame_order(self):
        inverse = np.array([[1., 0., 10.], [0., 1., 20.], [0., 0., 1.]])
        bottom, top, heatmap = self.positions(inverse=inverse)
        np.testing.assert_allclose(bottom, np.tile([40, 80], (9, 1)), atol=1e-10)
        np.testing.assert_allclose(top, np.tile([100, 50], (9, 1)), atol=1e-10)
        self.assertEqual(heatmap.sum(), 18)

    def test_leading_interior_and_trailing_missing_frames_stay_missing(self):
        _, top, heatmap = self.positions(missing=[0, 4, 8])
        np.testing.assert_array_equal(np.flatnonzero(np.isnan(top).all(axis=1)), [0, 4, 8])
        self.assertTrue(np.isfinite(top[[1, 2, 3, 5, 6, 7]]).all())
        self.assertEqual(top.shape, (9, 2))
        self.assertEqual(heatmap.sum(), 15)

    def test_missing_interior_frame_does_not_erase_neighbors(self):
        _, top, heatmap = self.positions(missing=[4])
        self.assertTrue(np.isnan(top[4]).all())
        np.testing.assert_allclose(top[[0, 1, 2, 3, 5, 6, 7, 8]], np.tile([90, 30], (8, 1)), atol=1e-10)
        self.assertEqual(heatmap.sum(), 17)

    def test_all_missing_top_frames_do_not_crash_or_add_heatmap_samples(self):
        bottom, top, heatmap = self.positions(missing=range(9))
        self.assertTrue(np.isfinite(bottom).all())
        self.assertTrue(np.isnan(top).all())
        self.assertEqual(heatmap.sum(), 9)

    def test_observed_origin_is_not_confused_with_a_missing_sample(self):
        _, top, heatmap = self.positions(top_box=(-10, -10, 10, 0))
        np.testing.assert_allclose(top, np.zeros((9, 2)), atol=1e-10)
        self.assertTrue(np.isfinite(top).all())
        self.assertEqual(heatmap.sum(), 18)


if __name__ == '__main__':
    unittest.main()
