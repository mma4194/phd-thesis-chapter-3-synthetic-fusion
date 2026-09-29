"""Compare freshly computed results with the approved paper reference."""
from pathlib import Path
import json,copy,hashlib
import pandas as pd
from workflow import ROOT,write_json
from reference_reader import Reader,compare,digest

def evaluate(run):
 run=Path(run);out=run/'paper_comparison';out.mkdir(exist_ok=True)
 path=ROOT/'reference/current/paper_reference_v1.json';reference=json.loads(path.read_text())
 adapted=copy.deepcopy(reference)
 source='residential_BINARY_V6/reports/cell17_4_baseline_metric_by_artifact.csv'
 for s in adapted['checks']:
  if s.get('root')=='evaluation':
   s['root']='run';s['path']=source
   if s.get('where',{}).get('baseline')=='CTGAN_numeric_counts_v1':s['where']['baseline']='CTGAN'
 reader=Reader(adapted,run,run);rows=[]
 for s in adapted['checks']:
  r={'id':s['id'],'expected':json.dumps(s['expected']),'actual':'','paper_precision_match':False,'strict_numeric_match':False,'detail':''}
  try:
   v=reader.get(s['id']);display,strict=compare(v,s['expected'],s['decimals'])
   r.update(actual=json.dumps(v),paper_precision_match=display,strict_numeric_match=strict,status='PASS' if display and strict else 'DIFFERENT')
  except (FileNotFoundError,ImportError) as e:r.update(status='BLOCKED',detail=str(e))
  except Exception as e:r.update(status='ERROR',detail=repr(e))
  rows.append(r)
 pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
 pd.DataFrame([r for r in rows if r['status']!='PASS'],columns=rows[0].keys()).to_csv(out/'issues.csv',index=False)
 changed=[p for p,f in reader.fingerprints.items() if f['kind']=='file_sha256' and digest(Path(p))!=f['sha256']]
 fallback=run/'residential_BINARY_V6/reports/baseline_c2st_fallback_events.json'
 events=json.loads(fallback.read_text())['events'] if fallback.is_file() else None
 audit=pd.read_csv(run/'residential_BINARY_V6/reports/cell17_1_ctgan_baseline_run_audit.csv')
 ctgan_ok=len(audit)==3 and audit.status.eq('success').all()
 complete=all(r['status']=='PASS' for r in rows) and not changed and events==[] and bool(ctgan_ok)
 result={'status':'PASS_FOR_VERSIONED_REFERENCE' if complete else 'DIFFERENT_OR_INCOMPLETE','reference_version':reference['reference_version'],'reference_sha256':digest(path),'checks':len(rows),'passed':sum(r['status']=='PASS' for r in rows),'different':sum(r['status']=='DIFFERENT' for r in rows),'blocked_or_errors':sum(r['status'] in ['BLOCKED','ERROR'] for r in rows),'all_ctgan_scopes_succeeded':bool(ctgan_ok),'baseline_C2ST_fallback_events':events,'inputs_changed_during_check':changed,'exhaustive_manuscript_coverage':False,'independent_run_repeatability_verified':False,'adapter':'Original immutable reference; CTGAN_numeric_counts_v1 maps to integrated CTGAN in the same two scopes. Evidence comes from the current run metric table.'}
 write_json(out/'SUMMARY.json',result);write_json(out/'input_fingerprints.json',reader.fingerprints)
 return {'paper_agreement':result['status'],'comparison':result}
