import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import configuration

spec = importlib.util.spec_from_file_location('residential_data', ROOT / 'tools/residential_data.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class ReviewerInputs(unittest.TestCase):
    def test_notebook_hash_is_independent_of_windows_text_encoding(self):
        from workflow import notebook_source_sha256
        notebook = ROOT / 'notebooks/Synthetic_Fusion_Chapter3.ipynb'
        expected = notebook_source_sha256(notebook)
        original = Path.read_text
        def windows_read(path, encoding=None, errors=None):
            return original(path, encoding=encoding or 'cp1252', errors=errors)
        with patch.object(Path, 'read_text', windows_read):
            self.assertEqual(notebook_source_sha256(notebook), expected)

    def test_local_config_is_shared_and_explicit_config_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config').mkdir()
            (root / 'config/local.json').write_text(json.dumps({'toniot_root': '/local/data'}))
            other = root / 'other.json'
            other.write_text(json.dumps({'toniot_root': '/other/data'}))
            with patch.object(configuration, 'ROOT', root):
                self.assertEqual(configuration.load_config()['toniot_root'], '/local/data')
                self.assertEqual(configuration.load_config(other)['toniot_root'], '/other/data')

    def test_local_residential_asset_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = root / 'datasets/residential/cps_unified_1s_FULLGRID_FINAL_WITH_IOT_STATEFUL_STRICTMASKED_TRIMMED.parquet'
            p.parent.mkdir(parents=True)
            p.write_bytes(b'input location fixture')
            with patch.object(configuration, 'ROOT', root):
                self.assertEqual(configuration.default_config()['residential_parquet'], str(p))

    def test_residential_identity_and_shape_are_both_required(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'input.parquet'
            pq.write_table(pa.table({'a': [1, 2], 'b': [3, 4]}), p)
            descriptor = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'rows': 2, 'physical_columns': 2}
            self.assertEqual(data.verify_file(p, descriptor)['status'], 'PASS')
            with self.assertRaises(ValueError):
                data.verify_file(p, dict(descriptor, sha256='0' * 64))
            with self.assertRaises(ValueError):
                data.verify_file(p, dict(descriptor, rows=3))

    def test_download_does_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'input.parquet'
            p.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                data.download('https://example.org/data.parquet', p, {})
            self.assertEqual(p.read_bytes(), b'preserve')


if __name__ == '__main__':
    unittest.main()
