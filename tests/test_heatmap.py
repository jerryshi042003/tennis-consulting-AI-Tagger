"""Exercise the production heatmap without importing unrelated detector models."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


source = Path(__file__).resolve().parents[1] / 'statistics_module.py'
tree = ast.parse(source.read_text())
stats = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Statistics')
method = next(node for node in stats.body if isinstance(node, ast.FunctionDef) and node.name == 'get_player_position_heatmap')
scope = {'np': np}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
heatmap = scope['get_player_position_heatmap']


class HeatmapTests(unittest.TestCase):
    def tracker(self, bottom, top=()):
        court = SimpleNamespace(court_total_width=160, court_total_height=160)
        return SimpleNamespace(court_tracker=SimpleNamespace(court_reference=court), feet_bottom=bottom, feet_top=top)

    def test_negative_samples_stay_on_the_nearest_edge(self):
        tracker = self.tracker([(-1, 80), (80, -1), (-1000, 0)], [(1000, 80)])
        np.testing.assert_array_equal(heatmap(tracker), [[1, 1], [1, 1]])

    def test_missing_samples_do_not_break_valid_tracking(self):
        tracker = self.tracker([(np.nan, 0), (0, np.inf), (1, 1)], [(80, 80)])
        np.testing.assert_array_equal(heatmap(tracker), [[1, 0], [0, 1]])

    def test_empty_tracking_returns_an_empty_grid(self):
        np.testing.assert_array_equal(heatmap(self.tracker([])), np.zeros((2, 2)))


if __name__ == '__main__':
    unittest.main()
