"""Execute a single documented stage in a clean process."""
from pathlib import Path
import argparse, importlib.util, inspect, json, os, shutil, sys, time, traceback, types
from workflow import ROOT, sha256, write_json
from determinism import enforce

def session_namespace():
 # Match a notebook kernel's importable __main__ namespace for dataclasses/pickles.
 from IPython.display import display
 module=types.ModuleType('__main__');module.__file__=str(ROOT/'notebooks/Synthetic_Fusion_Chapter3.ipynb')
 module.__dict__['display']=display
 sys.modules['__main__']=module
 return module.__dict__

def exec_cell(group,index,ns,run):
 p=ROOT/'source_cells'/group/(f'{index:03d}.py' if isinstance(index,int) else index)
 code=p.read_text();entry={'group':group,'cell':index,'sha256':sha256(p),'status':'RUNNING'}
 trace=run/'cell_execution.jsonl'
 with trace.open('a') as f:f.write(json.dumps(entry)+'\n')
 print(f'[{group} cell {index}] START',flush=True);t=time.monotonic()
 had_file='__file__' in ns;previous_file=ns.get('__file__')
 ns['__file__']=str(p)
 try:exec(compile(code,str(p),'exec',dont_inherit=True),ns)
 except BaseException:
  entry.update(status='FAILED',seconds=time.monotonic()-t)
  with trace.open('a') as f:f.write(json.dumps(entry)+'\n')
  raise
 finally:
  if had_file:ns['__file__']=previous_file
  else:ns.pop('__file__',None)
 entry.update(status='COMPLETED',seconds=time.monotonic()-t)
 with trace.open('a') as f:f.write(json.dumps(entry)+'\n')
 print(f'[{group} cell {index}] COMPLETED ({entry["seconds"]:.1f}s)',flush=True)

def setup_env(c,run):
 res=run/'residential_BINARY_V6'
 os.environ.update(CPS_INPUT_PARQUET=str(c.get('residential_parquet') or ''),CPS_OUTDIR=str(res),CPS_STUDY_FINAL_OUTDIR=str(res),CPS_BINARY_V6_OUTDIR=str(res),CPS_CANONICAL_RUN=str(res),SMARTSTAR_ARTIFACT_ROOT=str(run/'smartstar'),CPS_REAL_SPLIT_DIR=str(res/'synthetic'))
 return res

def load_public():
 sys.path.insert(0,str(ROOT/'public/src'))
 spec=importlib.util.spec_from_file_location('public_runner',ROOT/'public/run.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def run_public(stage,c,run):
 from argparse import Namespace
 mod=load_public();out=run/stage;out.mkdir()
 result=mod.run_public(Namespace(command=stage,input=c[stage+'_root'],timeout=int(c.get('public_timeout_seconds',0))),out)
 if result['status'] not in {'PASS','DIFFERENT'}:raise RuntimeError(json.dumps(result))
 if stage=='toniot':result['clock_check']=toniot_clock_check(out)
 return {'paper_agreement':'PASS_FOR_DOCUMENTED_PUBLIC_TABLES' if result['status']=='PASS' else 'DIFFERENT_FROM_PUBLIC_REFERENCE','comparison':result}

def toniot_clock_check(out):
 import verify
 result=verify.toniot_check(Path(out),actual=Path(out))
 result['scope']='Q4 and clock sensitivity from this run\'s aligned matrices, selected pairs and Q6 table'
 return result


def residential(c,run):
 import pandas as pd
 res=setup_env(c,run)
 if res.exists():raise RuntimeError('Residential target exists. Use a new run to prevent stale outputs.')
 from manifest_validation import validate_manifest
 validation=validate_manifest(ROOT)
 write_json(run/'q4_manifest_validation.json',validation)
 manifest=ROOT/'design_inputs';target=run/'residential_BINARY_V6_manifest';target.mkdir()
 copied=[]
 for p in sorted(manifest.glob('coupling_manifest*')):
  if p.is_file():
   shutil.copy2(p,target/p.name)
   if sha256(target/p.name)!=sha256(p):raise RuntimeError('Manifest copy identity failed: '+p.name)
   copied.append({'file':p.name,'sha256':sha256(p)})
 write_json(run/'q4_input_manifest.json',{'source':str(manifest),'files':copied,'input_type':'frozen_data_derived_train_val_manifest','upstream_regenerated':False,'validation_status':validation['status']})
 ns=session_namespace()
 for j in range(1,208):
  p=ROOT/'source_cells/residential'/f'{j:03d}.py'
  if not p.exists() and j != 103:continue
  if j==51:continue
  if j==103:
   # The historical cell searches unrelated directories by mtime and can write an empty sentinel.
   # Use the explicit historical manifest input instead. Pair selection remains in the next cell.
   ns['MANIFEST_DIR']=str(target)
   frames=[pd.read_csv(p) if p.suffix=='.csv' else pd.read_parquet(p) for p in target.glob('coupling_manifest*') if p.suffix in ['.csv','.parquet']]
   if not frames or not any(len(f)>0 and {'anchor_name','signal','manifest_tier','lag_lo','lag_hi'}.issubset(f.columns) for f in frames):raise ValueError('Invalid Q4 manifest schema or empty pairs')
   write_json(res/'artifacts/contracts/cell135_pre_manifest_recovery_contract.json',{'manifest_mode':'explicit_historical_input','source':str(manifest),'files':copied})
   continue
  if j==116:
   base=ns.get('PROTOCOL_SYN_TEST_130')
   if not isinstance(base,pd.DataFrame):raise RuntimeError('A0 pre-materialization protocol matrix missing')
   dest=res/'artifacts/A0_PROTOCOL_BASE_BEFORE_MATERIALIZATION.parquet'
   dest.parent.mkdir(parents=True,exist_ok=True);base.to_parquet(dest)
   write_json(run/'a0_runtime_state.json',{'captured_before_source_cell':116,'path':str(dest),'sha256':sha256(dest),'shape':list(base.shape),'historical_input_identity_verified':False})
  exec_cell('residential',j,ns,run)
  if j==53:
   if int(ns.get('CELL12C2_CACHE_HIT_N',-1))!=0:raise RuntimeError('Previous candidate reuse detected')
   metrics=pd.read_csv(res/'reports/cell12c2_all_val_candidate_metrics.csv')
   hits=metrics.get('cache_hit',pd.Series(False,index=metrics.index)).astype(str).str.lower().isin(['true','1']).sum()
   if hits:raise RuntimeError('Candidate output records indicate reuse')
   write_json(run/'fresh_continuous_training.json',{'source_cell':53,'candidate_rows':len(metrics),'previous_candidate_hits':int(hits),'previous_models_imported':False,'generated_in_this_run':True,'execution_mode':ns.get('CELL12C2_MODE')})
  if j==57:
   audit=pd.read_csv(res/'reports/cell12c4R_trainval_cache_materialization_audit.csv')
   if audit.fallback_used.astype(str).str.lower().isin(['true','1']).any():raise RuntimeError('Continuous materialization used an undeclared fallback')


 return {'paper_agreement':'NOT_TESTED','route':'fresh_training','meaning':'Every continuous candidate was regenerated. Compare outputs before claiming agreement.'}

def exec_selection_cell(index, ns, run):
 # This standalone diagnostic communicates through files, not shared globals.
 # Isolate its imports, helpers and working variables from the other cells.
 if index == 6:
  isolated = {'__name__': '__main__'}
  if 'display' in ns:
   isolated['display'] = ns['display']
  return exec_cell('selection', index, isolated, run)
 return exec_cell('selection', '016_fixed_policy.py' if index==16 else index, ns, run)

def post(stage,c,run):
 res=setup_env(c,run);ns=session_namespace()
 ids=range(3,9) if stage=='postrun' else [*range(2,14),16]
 group='postrun' if stage=='postrun' else 'selection'
 # Set each Fano policy exactly as implemented in the paper source; no additional seeds.
 os.environ['THESIS_RUN_Q5_EXTENDED_SEEDS']='0'
 for j in ids:
  if group=='selection':exec_selection_cell(j,ns,run)
  else:exec_cell(group,j,ns,run)
 if stage=='selection':
  import pandas as pd
  import pyarrow.parquet as pq
  target=res/'synthetic/CPS_SYNTHETIC_TEST_PUBLIC_Q6_CONSERVATIVE.parquet'
  reference=pd.read_csv(ROOT/'reference/recovered/q6_conservative_public_column_registry.csv')
  key=next(k for k in ['col','column','feature'] if k in reference)
  actual=pq.ParquetFile(target).schema_arrow.names
  actual=[x for x in actual if not x.startswith('__index_level_')]
  expected=reference[key].astype(str).tolist()
  ok=len(actual)==len(expected)==23 and set(actual)==set(expected)
  write_json(run/'candidate_schema_check.json',{'status':'PASS' if ok else 'DIFFERENT','actual_columns':actual,'expected_columns':expected})
  return {'paper_agreement':'PASS_FOR_CANDIDATE_SCHEMA_ONLY' if ok else 'DIFFERENT'}
 return {'paper_agreement':'NOT_TESTED'}

def inspection(c,run):
 import pandas as pd
 ns=session_namespace()
 # Configuration source is documentary; replace only its paths and selected inspection scope.
 exec_cell('inspection',3,ns,run)
 ns['CFG_INSPECT'].update(data_path=Path(c['residential_parquet']),run_root=run/'residential_BINARY_V6',output_parent=run/'inspection',scope='m7',print_every_feature=False,compute_source_sha256=bool(c.get('hash_inputs',True)))
 for j in [5,7,8,9,10,11,13,15,17]:exec_cell('inspection',j,ns,run)
 out=Path(ns['OUTPUT_DIR']);roles=pd.read_csv(ROOT/'reference/author_feature_roles.csv');counts=roles.role.value_counts().to_dict()
 expected={'iot_continuous':94,'iot_binary':19,'iot_observability':1,'iot_driver':1}
 # Role enum spelling is inherited; inspect explicit values rather than infer semantics from data.
 if len(roles)!=115 or roles.feature_name.duplicated().any() or counts!=expected:raise ValueError('Author-role input inventory')
 roles.to_csv(out/'author_feature_roles.csv',index=False);write_json(out/'author_role_counts.json',counts)
 return {'paper_agreement':'NOT_TESTED','output':str(out),'author_role_counts':counts,'manual_labels_are_inputs':True}

def regression(c,run):
 import numpy as np,pandas as pd
 from sklearn.ensemble import RandomForestRegressor
 from sklearn.metrics import mean_absolute_error,r2_score
 from q5_design import corrected_design
 res=run/'residential_BINARY_V6';out=run/'protocol_regression';out.mkdir()
 rp=res/'synthetic/REAL_TEST_SPLIT.parquet';sp=res/'synthetic/CPS_SYNTHETIC_TEST_PUBLIC_Q6_CONSERVATIVE.parquet'
 syn=pd.read_parquet(sp);real=pd.read_parquet(rp,columns=list(syn.columns))
 xr,yr,xs,ys,persist,fit,ev,starts,scope,meta=corrected_design(real,syn)
 write_json(out/'protocol_before_fit.json',dict(meta,seeds=list(range(20260710,20260715)),input_sha256={'real':sha256(rp),'synthetic':sha256(sp)}))
 scope.to_csv(out/'feature_scope.csv',index=False)
 pred=pd.DataFrame({'sample_start_row':starts[ev],'target':yr[ev],'persistence':persist[ev],'fit_mean':float(yr[fit].mean()),'fit_median':float(np.median(yr[fit]))});rows=[]
 def score(label,yhat,seed):
  mae=mean_absolute_error(yr[ev],yhat);rows.append({'model':label,'seed':seed,'mae':mae,'normalized_mae':mae/meta['target_scale'],'r2':r2_score(yr[ev],yhat)})
 for label in ['persistence','fit_mean','fit_median']:score(label,pred[label],None)
 for seed in range(20260710,20260715):
  for label,x,y in [('TRTR',xr,yr),('TSTR_reference_assisted',xs,ys)]:
   model=RandomForestRegressor(n_estimators=300,max_depth=6,min_samples_leaf=5,n_jobs=1,random_state=seed).fit(x[fit],y[fit]);yp=model.predict(xr[ev]);pred[f'{label}_{seed}']=yp;score(label,yp,seed)
  print('Paired learner seed completed:',seed,flush=True)
 scores=pd.DataFrame(rows);scores.to_csv(out/'q5_corrected_scores.csv',index=False);pred.to_csv(out/'q5_corrected_predictions.csv',index=False)
 a=scores[scores.model=='TRTR'].set_index('seed').normalized_mae;b=scores[scores.model=='TSTR_reference_assisted'].set_index('seed').normalized_mae;loss=b-a
 summary={'mean_signed_loss':loss.mean(),'sd_signed_loss':loss.std(ddof=1),'median_absolute_gap':loss.abs().median(),'p95_absolute_gap':loss.abs().quantile(.95),'fit_samples':len(fit),'eval_samples':len(ev),'target_scale':meta['target_scale']}
 write_json(out/'summary.json',summary)
 current_ref=ROOT/'reference/current/q5_scores.csv'
 import verify
 try:
  verify.compare_csv(out/'q5_corrected_scores.csv',current_ref,['model','seed'],rtol=1e-7,atol=1e-9)
  match=True;detail=''
 except ValueError as e:match=False;detail=str(e)
 write_json(out/'reference_comparison.json',{'reference_version':'2026-09-29-paper-v1','matches_current_scores':match,'detail':detail})
 return {'paper_agreement':'PASS_FOR_CURRENT_REGRESSION' if match else 'DIFFERENT_FROM_CURRENT_REGRESSION'}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--stage',required=True);a=ap.parse_args();run=a.run;c=json.loads((run/'config.json').read_text());status=run/'status'/f'{a.stage}.json'
 try:
  enforce()
  if a.stage=='controlled':
   load_public();import verify
   out=run/'controlled_stage';out.mkdir();r={'paper_agreement':'PASS_FOR_CONTROLLED_TRIALS','checks':verify.controlled_check(out)}
  elif a.stage in ['smartstar','toniot']:r=run_public(a.stage,c,run)
  elif a.stage=='residential':r=residential(c,run)
  elif a.stage in ['postrun','selection']:r=post(a.stage,c,run)
  elif a.stage=='inspection':r=inspection(c,run)
  elif a.stage=='protocol_regression':
   load_public();r=regression(c,run)
  elif a.stage=='window_reconstruction':
   from window_reconstruction import evaluate_windows
   r=evaluate_windows(run/'residential_BINARY_V6',run/'window_reconstruction')
  elif a.stage=='paper_comparison':
   from current_reference import evaluate
   r=evaluate(run)
  elif a.stage=='figures':
   from figures import generate
   r=generate(run)
  elif a.stage=='repeatability_inventory':
   from repeatability import inventory
   r=inventory(run)
   if not r["inventory_complete"]:raise RuntimeError("Inventory incomplete; inspect repeatability_inventory.json")
  else:raise ValueError(a.stage)
  write_json(status,{'stage':a.stage,'execution_status':'COMPLETED',**r})
 except BaseException as e:
  traceback.print_exc();write_json(status,{'stage':a.stage,'execution_status':'FAILED','paper_agreement':'NOT_TESTED','error':str(e)});raise
if __name__=='__main__':main()
