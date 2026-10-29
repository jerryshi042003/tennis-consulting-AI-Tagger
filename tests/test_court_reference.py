"""Native checks for generated court references; no detector weights required."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


HAS_NATIVE_LIBRARIES = all(
    importlib.util.find_spec(name) is not None
    for name in ('cv2', 'numpy', 'matplotlib')
)


@unittest.skipUnless(HAS_NATIVE_LIBRARIES, 'requires OpenCV, NumPy and Matplotlib')
class CourtReferenceTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        from court_reference import CourtReference

        self.cv2 = cv2
        self.np = np
        self.reference_class = CourtReference
        self.folder = tempfile.TemporaryDirectory()
        self.previous_directory = Path.cwd()
        os.chdir(self.folder.name)
        self.addCleanup(self.folder.cleanup)
        self.addCleanup(os.chdir, self.previous_directory)

    def test_fresh_constructor_builds_loaded_scale_without_writing(self):
        reference = self.reference_class()
        self.assertEqual(reference.court.shape, (3506, 1665))
        self.assertEqual(reference.court.dtype, self.np.uint8)
        self.assertEqual(set(self.np.unique(reference.court)), {0, 255})
        self.assertEqual(reference.court[561, 286], 255)
        self.assertEqual(reference.court[1748, 1000], 0)  # net is not drawn
        self.assertEqual(list(Path('.').iterdir()), [])
        warped = self.cv2.warpPerspective(
            reference.court, self.np.eye(3), (1665, 3506)
        )
        self.np.testing.assert_array_equal(warped, reference.court)

    def test_existing_reference_is_loaded_and_preserved(self):
        path = Path('court_configurations/court_reference.png')
        path.parent.mkdir()
        custom = self.np.arange(12, dtype=self.np.uint8).reshape(3, 4)
        self.assertTrue(self.cv2.imwrite(str(path), custom))
        original = path.read_bytes()
        reference = self.reference_class()
        self.np.testing.assert_array_equal(reference.court, custom)
        self.assertEqual(path.read_bytes(), original)

    def test_unreadable_reference_falls_back_without_overwriting(self):
        path = Path('court_configurations/court_reference.png')
        path.parent.mkdir()
        path.write_bytes(b'not an image')
        reference = self.reference_class()
        self.assertEqual(reference.court.shape, (3506, 1665))
        self.assertEqual(set(self.np.unique(reference.court)), {0, 255})
        self.assertEqual(path.read_bytes(), b'not an image')

    def test_explicit_builder_exports_to_missing_directory(self):
        reference = self.reference_class()
        loaded_scale = reference.court.copy()
        binary = reference.build_court_reference()
        self.assertEqual(set(self.np.unique(binary)), {0, 1})
        path = Path('court_configurations/court_reference.png')
        self.assertTrue(path.is_file())
        exported = self.cv2.imread(str(path), self.cv2.IMREAD_GRAYSCALE)
        self.np.testing.assert_array_equal(exported, loaded_scale)
        self.np.testing.assert_array_equal(binary * 255, loaded_scale)
        self.np.testing.assert_array_equal(self.reference_class().court, loaded_scale)

    def test_configuration_export_creates_its_directory(self):
        reference = self.reference_class()
        reference.save_all_court_configurations()
        outputs = sorted(Path('court_configurations').glob('court_conf_*.png'))
        self.assertEqual(len(outputs), 12)
        for path in outputs:
            self.assertIsNotNone(self.cv2.imread(str(path)))


if __name__ == '__main__':
    unittest.main()
