"""Boundary tests: a scientific mismatch or absent input must never pass."""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from verify import compare_csv
from controlled import Claim,Evidence,assess
class ArtifactTests(unittest.TestCase):
    def test_altered_measurement_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a.csv';b=Path(t)/'b.csv'
            pd.DataFrame({'id':['a','b'],'score':[1.,2.]}).to_csv(a,index=False)
            pd.DataFrame({'id':['a','b'],'score':[1.,3.]}).to_csv(b,index=False)
            with self.assertRaises(ValueError):compare_csv(a,b,['id'])
    def test_duplicate_identity_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a.csv';pd.DataFrame({'id':['a','a'],'score':[1.,1.]}).to_csv(a,index=False)
            with self.assertRaises(ValueError):compare_csv(a,a,['id'])
    def test_missing_public_input_is_not_pass(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'run'
            p=subprocess.run([sys.executable,str(ROOT/'run.py'),'toniot','--input',str(Path(t)/'absent'),'--output',str(out)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertEqual(json.loads((out/'status.json').read_text())['status'],'MISSING_INPUT')
    def test_missing_claim_record(self):
        c=Claim('a','scope','p',('x',));self.assertEqual(assess(c,[]).status,'not_assessed')
    def test_existing_run_is_protected(self):
        with tempfile.TemporaryDirectory() as t:
            marker=Path(t)/'keep.txt';marker.write_text('unchanged')
            p=subprocess.run([sys.executable,str(ROOT/'run.py'),'toniot','--output',t],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertEqual(marker.read_text(),'unchanged')
if __name__=='__main__':unittest.main()
