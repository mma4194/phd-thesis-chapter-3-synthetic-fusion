import json, shutil, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from manifest_validation import validate_manifest

class ManifestTests(unittest.TestCase):
    def test_archived_records_agree(self):
        r = validate_manifest(ROOT)
        self.assertEqual(r['status'], 'PASS')
        self.assertEqual(len(r['checks']), 52)
        self.assertFalse(r['upstream_alignment_regenerated'])
    def test_modified_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shutil.copytree(ROOT / 'design_inputs', root / 'design_inputs')
            p = root / 'design_inputs/coupling_manifest_tierA.csv'
            with p.open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                validate_manifest(root)
    def test_missing_supporting_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shutil.copytree(ROOT / 'design_inputs', root / 'design_inputs')
            (root / 'design_inputs/supporting_records/alignment_results_train.csv').unlink()
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                validate_manifest(root)
    def test_eligibility_is_checked_beyond_hashes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shutil.copytree(ROOT / 'design_inputs', root / 'design_inputs')
            p = root / 'design_inputs/coupling_manifest_meta.json'
            meta = json.loads(p.read_text())
            meta['tierA_rules']['min_coverage_core'] = 1.1
            p.write_text(json.dumps(meta))
            lockpath = root / 'design_inputs/INPUT_LOCK.json'
            lock = json.loads(lockpath.read_text())
            for entry in lock['files']:
                if entry['path'] == p.name:
                    entry['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
            lockpath.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, 'tierA_coverage'):
                validate_manifest(root)

if __name__ == '__main__':
    unittest.main()
