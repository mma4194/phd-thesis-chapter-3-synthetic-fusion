import ast,contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ctgan_encoding import classify
import workflow,repeatability,worker
class IntegratedTests(unittest.TestCase):
 def test_numeric_counts_and_binary_categories(self):
  data=pd.DataFrame({'count':[0.,1.,0.],'event':[0.,1.,0.],'category':[3.,4.,3.]})
  policy={'numeric_count_columns':['count'],'event_indicator_columns_require_train_binary':['event'],'categorical_columns':['category']}
  self.assertEqual(classify(data,data,policy)[0],['event','category'])
  data.loc[0,'event']=2
  with self.assertRaises(ValueError):classify(data,data,policy)
 def test_unknown_and_invalid_counts(self):
  policy={'numeric_count_columns':['count'],'event_indicator_columns_require_train_binary':[],'categorical_columns':[]}
  for d in [pd.DataFrame({'unknown':[1]}),pd.DataFrame({'count':[-1.]}),pd.DataFrame({'count':[1.5]})]:
   with self.assertRaises(ValueError):classify(d,d,policy)
 def test_datetime_isolation(self):
  from datetime import datetime,timezone
  ns={'datetime':datetime,'timezone':timezone}
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);p=r/'source_cells/selection';p.mkdir(parents=True);(p/'006.py').write_text('import datetime\n')
   with patch.object(worker,'ROOT',r),contextlib.redirect_stdout(io.StringIO()):worker.exec_selection_cell(6,ns,r)
  self.assertIs(ns['datetime'],datetime)
 def test_metadata_only_and_mixed_objects_inventory(self):
  with tempfile.TemporaryDirectory() as d:
   run=Path(d);p=run/'residential_BINARY_V6/reports';p.mkdir(parents=True)
   (p/'metadata.csv').write_text('created_utc,path\n2026-01-01,/x\n');(p/'mixed.csv').write_text('value,nullable\nTrue,\nhello,3\n')
   self.assertFalse(repeatability.inventory(run)['inventory_complete'])
   inv=json.loads((run/'repeatability_inventory.json').read_text())
   self.assertEqual(inv['tables']['residential_BINARY_V6/reports/metadata.csv']['content_status'],'METADATA_ONLY_EXCLUDED');self.assertFalse(inv['table_errors'])
 def test_completed_stage_skips_and_changed_output_blocks(self):
  with tempfile.TemporaryDirectory() as d:
   run=Path(d);p=run/'controlled_stage/controlled/decision_by_setting.csv';p.parent.mkdir(parents=True);p.write_text('x\n1\n')
   workflow.write_json(run/'status/controlled.json',{'execution_status':'COMPLETED'});workflow.write_json(run/'receipts/controlled.json',workflow.primary_receipt(run,'controlled'))
   with patch.object(workflow,'validate_resume',return_value={}),patch.object(workflow.subprocess,'Popen') as popen,contextlib.redirect_stdout(io.StringIO()):
    workflow.run_stage(run,'controlled',resume=True);popen.assert_not_called();p.write_text('x\n2\n')
    with self.assertRaises(RuntimeError):workflow.run_stage(run,'controlled',resume=True)
    popen.assert_not_called()
 def test_failed_residential_not_replayed(self):
  with tempfile.TemporaryDirectory() as d:
   run=Path(d);workflow.write_json(run/'status/residential.json',{'execution_status':'FAILED'})
   with patch.object(workflow,'validate_resume',return_value={}),patch.object(workflow.subprocess,'Popen') as popen:
    with self.assertRaisesRegex(RuntimeError,'automatic in-stage replay'):workflow.run_stage(run,'residential',resume=True)
    popen.assert_not_called()
 def test_embedded_worker(self):
  tree=ast.parse((ROOT/'source_cells/residential/149.py').read_text())
  node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='worker_code' for t in n.targets))
  code=ast.literal_eval(node.value);compile(code,'worker','exec')
  for expected in ['model.set_random_state(seed)','model.save(','model.fit(train, discrete_columns=discrete_columns)']:self.assertIn(expected,code)
 def test_notebook_compiles(self):
  import nbformat
  nb=nbformat.read(ROOT/'notebooks/Synthetic_Fusion_Chapter3.ipynb',as_version=4);nbformat.validate(nb)
  for i,c in enumerate(nb.cells):
   if c.cell_type=='code':compile(c.source,f'cell_{i}','exec')
if __name__=='__main__':unittest.main()
