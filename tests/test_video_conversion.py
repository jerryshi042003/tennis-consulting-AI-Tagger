"""Exercise the production converter without importing unrelated detector models."""
import ast
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


source = Path(__file__).resolve().parents[1] / 'create_data.py'
method = next(node for node in ast.parse(source.read_text()).body
              if isinstance(node, ast.FunctionDef) and node.name == 'convert_avi_to_mp4')
scope = {'os': os, 'Path': Path, 'subprocess': subprocess, 'tempfile': tempfile}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
convert = scope['convert_avi_to_mp4']


class VideoConversionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.avi = self.root / 'original video ; $name.avi'
        self.avi.write_bytes(b'original AVI fixture')
        self.output = self.root / 'converted video ; $name.mp4'

    def assert_original_retained(self):
        self.assertEqual(self.avi.read_bytes(), b'original AVI fixture')
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.iterdir()), [self.avi])

    def test_failed_encoder_preserves_original_and_discards_partial_output(self):
        def encode(command, **kwargs):
            Path(command[-1]).write_bytes(b'partial MP4')
            raise subprocess.CalledProcessError(1, command)

        with patch.object(subprocess, 'run', side_effect=encode):
            with self.assertRaises(subprocess.CalledProcessError):
                convert(self.avi, self.output)
        self.assert_original_retained()

    def test_missing_encoder_preserves_original(self):
        with patch.object(subprocess, 'run', side_effect=FileNotFoundError('ffmpeg')):
            with self.assertRaises(FileNotFoundError):
                convert(self.avi, self.output)
        self.assert_original_retained()

    def test_missing_or_empty_output_preserves_original(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                def encode(command, **kwargs):
                    if empty:
                        Path(command[-1]).touch()

                with patch.object(subprocess, 'run', side_effect=encode):
                    with self.assertRaises(RuntimeError):
                        convert(self.avi, self.output)
                self.assert_original_retained()

    def test_existing_output_is_never_overwritten(self):
        self.output.write_bytes(b'existing output')
        with patch.object(subprocess, 'run') as encode:
            with self.assertRaises(FileExistsError):
                convert(self.avi, self.output)
            encode.assert_not_called()
        self.assertEqual(self.output.read_bytes(), b'existing output')
        self.assertEqual(self.avi.read_bytes(), b'original AVI fixture')

    def test_concurrent_output_is_never_overwritten(self):
        def encode(command, **kwargs):
            Path(command[-1]).write_bytes(b'new encoded output')
            self.output.write_bytes(b'concurrent output')

        with patch.object(subprocess, 'run', side_effect=encode):
            with self.assertRaises(FileExistsError):
                convert(self.avi, self.output)
        self.assertEqual(self.output.read_bytes(), b'concurrent output')
        self.assertEqual(self.avi.read_bytes(), b'original AVI fixture')
        self.assertEqual(set(self.root.iterdir()), {self.avi, self.output})

    def test_success_publishes_complete_output_before_removing_original(self):
        def encode(command, **kwargs):
            self.assertEqual(command[command.index('-i') + 1], str(self.avi))
            self.assertEqual(kwargs, {'check': True})
            self.assertNotIn(str(self.output), command)
            self.assertTrue(self.avi.exists())
            self.assertFalse(self.output.exists())
            Path(command[-1]).write_bytes(b'complete MP4 fixture')

        with patch.object(subprocess, 'run', side_effect=encode) as encode:
            self.assertTrue(convert(self.avi, self.output))
            self.assertIsInstance(encode.call_args.args[0], list)
        self.assertFalse(self.avi.exists())
        self.assertEqual(self.output.read_bytes(), b'complete MP4 fixture')
        self.assertEqual(list(self.root.iterdir()), [self.output])

    def test_failed_source_removal_retains_both_files(self):
        def encode(command, **kwargs):
            Path(command[-1]).write_bytes(b'complete MP4 fixture')

        with patch.object(subprocess, 'run', side_effect=encode):
            with patch.object(Path, 'unlink', side_effect=PermissionError('retained AVI')):
                with self.assertRaises(PermissionError):
                    convert(self.avi, self.output)
        self.assertEqual(self.avi.read_bytes(), b'original AVI fixture')
        self.assertEqual(self.output.read_bytes(), b'complete MP4 fixture')
        self.assertEqual(set(self.root.iterdir()), {self.avi, self.output})


if __name__ == '__main__':
    unittest.main()
