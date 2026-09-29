"""Exercise the actual metric functions on controlled inputs, without generators."""
import ast,builtins,gc,hashlib,json,os,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
class BaselineMetricTests(unittest.TestCase):
 def engine(self):
  tree=ast.parse((ROOT/'source_cells/residential/152.py').read_text())
  ns=dict(np=np,pd=pd,gc=gc,os=os,json=json,hashlib=hashlib,CFG={},HIST_BINS_174=50,LOW_CARD_UNIQUE_174=30,LAGS_174=[1,5,10,30,60],ACTIVITY_EPS_174=1e-12,C2ST_FALLBACK_EVENTS_174=[])
  exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'actual_metrics','exec'),ns)
  return ns
 def test_penalty_uses_declared_five_components(self):
  s=self.engine()['_artifact_score_tabular_174']({'mean_ks':.1,'mean_hist_tv':.2,'mean_wasserstein_norm':.3,'correlation_mae':.4,'c2st_auc':.9,'mean_support_jaccard':.99})
  self.assertAlmostEqual(s,.28)
 def test_identical_data_and_classifier_fallback_recording(self):
  ns=self.engine();d=pd.DataFrame(np.random.default_rng(123).poisson(3,size=(600,3)),columns=['a','b','c'])
  self.assertAlmostEqual(ns['_c2st_auc_proxy_174'](d,d,list(d)),.5)
  self.assertEqual(ns['C2ST_FALLBACK_EVENTS_174'],[])
  original=builtins.__import__
  def blocked(name,*args,**kwargs):
   if name.startswith('sklearn'):raise ImportError('test-only forced classifier import failure')
   return original(name,*args,**kwargs)
  with patch('builtins.__import__',side_effect=blocked):ns['_c2st_auc_proxy_174'](d,d,list(d))
  self.assertEqual(len(ns['C2ST_FALLBACK_EVENTS_174']),1)
if __name__=='__main__':unittest.main()
