"""Prevent metadata changes and missing evidence from giving misleading results."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from workflow import notebook_source_sha256, write_json
from repeatability import compare_runs, baseline_training_status


class RepeatabilityTests(unittest.TestCase):
    def fixture(self, run):
        run.mkdir()
        inventory = {
            'outputs': {'data.parquet': {'sha256_ordered_values': 'same'}},
            'tables': {'scores.csv': {'sha256_ordered_values': 'same'}},
            'input_inventories': {'smartstar': [{'sha256': 'a'}], 'toniot': [{'sha256': 'b'}]},
            'table_errors': {}, 'inventory_errors': {},
            'baseline_training': {'all_ctgan_scopes_succeeded': False},
        }
        objects = {'repeatability_inventory.json': inventory,
                   'environment.json': {'python': 'fixed'},
                   'environment_full.json': {'packages': {'numpy': 'fixed'}},
                   'hardware.json': {'gpu': 'fixed'},
                   'package_identity.json': {'source': 'fixed'},
                   'config.json': {'run_dir': str(run), 'seed': 12},
                   'FINAL_REPORT.json': {'all_stages_completed': True},
                   'fresh_continuous_training.json': {'previous_candidate_hits': 0}}
        for name, value in objects.items():
            write_json(run / name, value)

    def test_exact_outputs_do_not_imply_all_baselines_succeeded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root / 'a'); self.fixture(root / 'b')
            result = compare_runs(root / 'a', root / 'b', root / 'comparison')
            self.assertEqual(result['status'], 'PASS_FOR_LISTED_OUTPUTS')
            self.assertFalse(result['both_runs_all_ctgan_scopes_succeeded'])

    def test_missing_candidate_inventory_blocks_exact_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root / 'a'); self.fixture(root / 'b')
            path = root / 'b/repeatability_inventory.json'
            inventory = json.loads(path.read_text())
            inventory['inventory_errors']['candidate'] = 'missing'
            write_json(path, inventory)
            result = compare_runs(root / 'a', root / 'b', root / 'comparison')
            self.assertEqual(result['status'], 'DIFFERENT_OR_INCOMPLETE')

    def test_saved_outputs_do_not_change_notebook_source_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'notebook.ipynb'
            nb = {'cells': [{'cell_type': 'code', 'source': ['x = 1'],
                             'outputs': [], 'execution_count': None}]}
            write_json(path, nb); first = notebook_source_sha256(path)
            nb['cells'][0].update(outputs=[{'text': 'new output'}], execution_count=1)
            write_json(path, nb); self.assertEqual(first, notebook_source_sha256(path))
            nb['cells'][0]['source'] = ['x = 2']
            write_json(path, nb); self.assertNotEqual(first, notebook_source_sha256(path))

    def test_missing_baseline_audit_is_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(baseline_training_status(folder)['all_ctgan_scopes_succeeded'])


if __name__ == '__main__':
    unittest.main()
