import json,tempfile,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from reference_reader import Reader,compare

class CheckerTests(unittest.TestCase):
    def test_rounding_does_not_hide_drift(self):
        self.assertEqual(compare(.10000004,.10000001,3),(True,False))
    def test_pair_shape_and_nonfinite(self):
        with self.assertRaises(ValueError):compare([1],[1,2],[0,0])
        with self.assertRaises(ValueError):compare(float('nan'),1,3)
    def test_missing_evidence_is_not_zero(self):
        with tempfile.TemporaryDirectory() as t:
            r=Reader({'checks':[dict(id='a',op='csv',path='missing.csv',aggregate='rows')]},Path(t),Path(t))
            with self.assertRaises(FileNotFoundError):r.get('a')
    def test_duplicates_do_not_silently_select_first(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'data.csv').write_text('group,x\na,1\na,2\n')
            r=Reader({'checks':[dict(id='a',op='csv',path='data.csv',aggregate='single',column='x',where={'group':'a'})]},p,p)
            with self.assertRaises(ValueError):r.get('a')
    def test_changes_are_detected_from_actual_table(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'data.csv').write_text('group,x\na,2\nb,9\n')
            s={'checks':[dict(id='a',op='csv',path='data.csv',aggregate='single',column='x',where={'group':'a'})]}
            self.assertEqual(Reader(s,p,p).get('a'),2)
            (p/'data.csv').write_text('group,x\na,3\nb,9\n')
            self.assertEqual(compare(Reader(s,p,p).get('a'),2,0),(False,False))
    def test_logical_and_physical_parquet_columns(self):
        import pyarrow as pa,pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);pq.write_table(pa.table({'x':[1,2],'__index_level_0__':[0,1]}),p/'a.parquet')
            s={'checks':[dict(id=k,op='parquet',path='a.parquet',metric='columns') for k in ['PublicCols','PublicColsOnDisk']]}
            r=Reader(s,p,p);self.assertEqual(r.get('PublicCols'),1);self.assertEqual(r.get('PublicColsOnDisk'),2)

if __name__=='__main__':unittest.main()
