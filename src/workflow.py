"""Fresh-process orchestration for the complete Synthetic Fusion thesis workflow."""
from pathlib import Path
import csv,hashlib,importlib.metadata,json,os,platform,subprocess,sys,time,uuid
ROOT=Path(__file__).resolve().parents[1]
STAGES=['controlled','smartstar','toniot','residential','postrun','selection','inspection','protocol_regression','window_reconstruction','paper_comparison','figures','repeatability_inventory']
INPUT_SHA256='a4d7446daba0e46dab2dc21607a99c5a734a5366a6c24018a82ef2ef35fc4bda'
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)
def notebook_source_sha256(path):
 # Saving execution outputs must not change the identity of notebook source code.
 nb=json.loads(Path(path).read_text(encoding='utf-8'))
 cells=[{'cell_type':x['cell_type'],'source':''.join(x['source']) if isinstance(x['source'],list) else x['source']} for x in nb['cells']]
 return hashlib.sha256(json.dumps(cells,sort_keys=True).encode()).hexdigest()
def environment():
 packages={}
 for n in ['numpy','pandas','scipy','scikit-learn','pyarrow','torch','sdv','ctgan','copulas','rdt','deepecho','matplotlib','nbformat','nbclient','ipython']:
  try:packages[n]=importlib.metadata.version(n)
  except importlib.metadata.PackageNotFoundError:packages[n]='MISSING'
 return {'python':sys.version,'platform':platform.platform(),'packages':packages}
def new_run(config):
 c=dict(config)
 if int(c.get('n_jobs',1))!=1:raise ValueError('Use n_jobs=1 for the strict repeatability workflow.')
 for forbidden in ['continuous_cache_root','continuous_mode','allow_continuous_retraining','q4_manifest_root','reference_run_root']:
  if forbidden in c:raise ValueError(f'Obsolete configuration field: {forbidden}. Use the clean notebook configuration.')
 for key in ['residential_parquet','smartstar_root','toniot_root']:
  c[key]=str(Path(c[key]).expanduser().resolve())
 parent=Path(c['output_parent']).expanduser().resolve()
 if parent==ROOT or ROOT in parent.parents:raise ValueError('Choose an output folder outside the package, preferably scratch.')
 out=parent/(time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]);out.mkdir(parents=True,exist_ok=False)
 c.update(run_dir=str(out),continuous_training='fresh',reuse_previous_models=False)
 write_json(out/'config.json',c);write_json(out/'environment.json',environment())
 full={d.metadata.get('Name','unknown'):d.version for d in importlib.metadata.distributions()}
 write_json(out/'environment_full.json',{'python_executable':sys.executable,'packages':dict(sorted(full.items()))})
 (out/'environment.lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(full.items()))+'\n')
 hardware={'cpu':platform.processor(),'platform':platform.platform()}
 try:
  hardware['cpu_model']=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name'))
 except Exception:pass
 try:
  hardware['gpu_driver']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True).stdout.strip().splitlines()
 except Exception as e:hardware['gpu_driver_unavailable']=type(e).__name__
 try:
  import torch
  hardware.update(torch_cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version(),cuda_available=torch.cuda.is_available(),gpu_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
 except Exception as e:hardware['torch_error']=str(e)
 write_json(out/'hardware.json',hardware)
 write_json(out/'package_identity.json',{'manifest_sha256':sha256(ROOT/'MANIFEST.json'),'version':'2.1.0-chapter3-reviewer','notebook_source_sha256':notebook_source_sha256(ROOT/'notebooks/Synthetic_Fusion_Chapter3.ipynb')})
 return out

def preflight(run):
 run=Path(run);c=json.loads((run/'config.json').read_text());rows=[]
 def add(stage,name,ok,detail):rows.append(dict(stage=stage,check=name,status='PASS' if ok else 'BLOCKED',detail=str(detail)))
 env=environment();expected=json.loads((ROOT/'environment/validated_versions.json').read_text())
 add('all','python_version',platform.python_version()==expected['python'],platform.python_version())
 for n,v in expected['packages'].items():
  got=env['packages'].get(n,'MISSING');add('all','version:'+n,got.split('+')[0]==v.split('+')[0],f'installed={got}; expected={v}')
 raw=Path(c['residential_parquet']);add('residential','source_parquet',raw.is_file(),raw)
 if raw.is_file():
  try:
   import pyarrow.parquet as pq
   q=pq.ParquetFile(raw);add('residential','source_shape',q.metadata.num_rows==1277694 and len(q.schema_arrow.names)==740,f'{q.metadata.num_rows} x {len(q.schema_arrow.names)}')
   h=sha256(raw);add('residential','source_sha256',h==INPUT_SHA256,h)
  except Exception as e:add('residential','source_parquet_read',False,e)
 for x in ['circuit','environmental','switch','furnace','door','motion']:
  p=Path(c['smartstar_root'])/('homeA-'+x);add('smartstar','homeA-'+x,p.is_dir() and any(p.rglob('*')),p)
 tr=Path(c['toniot_root'])/'Processed_datasets'
 names=[f'Processed_Network_dataset/Network_dataset_{i}.csv' for i in range(1,24)]+[f'Processed_IoT_dataset/IoT_{s}.csv' for s in ['Fridge','GPS_Tracker','Garage_Door','Modbus','Motion_Light','Thermostat','Weather']]
 for n in names:add('toniot',n,(tr/n).is_file(),tr/n)
 add('residential','fresh_training_source',(ROOT/'source_cells/residential/053.py').is_file(),'All continuous candidates are trained in this run.')
 try:
  from manifest_validation import validate_manifest
  result=validate_manifest(ROOT)
  write_json(run/'q4_manifest_validation.json',result)
  add('residential','frozen_q4_manifest',result['status']=='PASS',f"{len(result['checks'])} identity, consistency and eligibility checks; upstream alignment is a supplied input")
 except Exception as e:
  write_json(run/'q4_manifest_validation.json',{'status':'FAIL','error':str(e)})
  add('residential','frozen_q4_manifest',False,e)
 from package_integrity import verify
 bad=verify(ROOT);add('all','package_integrity',not bad,bad if bad else 'All distributed files match their recorded hashes.')
 h=json.loads((run/'hardware.json').read_text())
 add('residential','cuda_available',h.get('cuda_available',False),h.get('gpu_devices',[]))
 add('residential','gpu_reference_class',all(x=='NVIDIA L40S' for x in h.get('gpu_devices',[])) and bool(h.get('gpu_devices')),h.get('gpu_devices',[]))
 add('residential','cuda_build',h.get('torch_cuda_build')=='12.8',h.get('torch_cuda_build'))
 add('residential','cudnn_build',h.get('cudnn')==91002,h.get('cudnn'))
 write_json(run/'preflight.json',rows)
 with (run/'preflight.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['stage','check','status','detail']);w.writeheader();w.writerows(rows)
 return rows

def process_environment():
 env=os.environ.copy()
 for k in list(env):
  if k.startswith(('CPS_','THESIS_','SMARTSTAR_','TONIOT_')):env.pop(k)
 env.update(PYTHONHASHSEED='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',CUBLAS_WORKSPACE_CONFIG=':4096:8',CPS_DETERMINISM_MODE='strict')
 return env

# Completion receipts describe the primary generated outputs used downstream.
REQUIRED_OUTPUTS={
 'controlled':['controlled_stage/controlled/decision_by_setting.csv'],
 'smartstar':['smartstar/ledgers/test_evidence_q1_q2_q3_ledger.csv','smartstar/manifests/q6_public_source_release_manifest.csv','smartstar/input_inventory.json'],
 'toniot':['toniot/intermediate/toniot_aligned_windows_with_split.parquet','toniot/physical_q4/tables/toniot_transfer_claim_scope_summary_physical_q4.csv','toniot/input_inventory.json'],
 'residential':['residential_BINARY_V6/synthetic/CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet','residential_BINARY_V6/synthetic/REAL_TEST_SPLIT.parquet','residential_BINARY_V6/reports/cell17_4_baseline_metric_by_artifact.csv','fresh_continuous_training.json'],
 'postrun':['residential_BINARY_V6/reports/review_fix_outputs/window_vs_pooled_reconciliation.csv'],
 'selection':['residential_BINARY_V6/synthetic/CPS_SYNTHETIC_TEST_PUBLIC_Q6_CONSERVATIVE.parquet','candidate_schema_check.json'],
 'protocol_regression':['protocol_regression/q5_corrected_predictions.csv','protocol_regression/q5_corrected_scores.csv','protocol_regression/summary.json'],
 'window_reconstruction':['window_reconstruction/window_metrics.csv'],
 'paper_comparison':['paper_comparison/comparison.csv','paper_comparison/SUMMARY.json'],
 'figures':['figures/fig02_controlled_decisions.pdf','figures/fig03_empirical_quality.pdf','figures/figS1_candidate_filtering.pdf'],
 'repeatability_inventory':['repeatability_inventory.json'],
}

def primary_receipt(run,stage):
 paths=list(REQUIRED_OUTPUTS.get(stage,[]))
 if stage=='inspection':
  paths=[str(p.relative_to(run)) for p in (run/'inspection').rglob('author_role_counts.json')]
  if not paths:raise RuntimeError('Inspection output missing')
 if not paths:raise RuntimeError('No declared completion outputs: '+stage)
 result={}
 for name in paths:
  p=run/name
  if not p.is_file():raise FileNotFoundError('Missing stage output: '+str(p))
  result[name]={'sha256':sha256(p),'bytes':p.stat().st_size}
 return result

def validate_resume(run):
 run=Path(run).resolve();c=json.loads((run/'config.json').read_text())
 if Path(c['run_dir']).resolve()!=run:raise ValueError('Run was moved. Keep its original path for resuming.')
 identity=json.loads((run/'package_identity.json').read_text())
 if identity['manifest_sha256']!=sha256(ROOT/'MANIFEST.json'):raise ValueError('Package changed; resume requires the same integrated package.')
 if identity['notebook_source_sha256']!=notebook_source_sha256(ROOT/'notebooks/Synthetic_Fusion_Chapter3.ipynb'):raise ValueError('Notebook source changed since the run. Use the unchanged notebook and external config file.')
 recorded=json.loads((run/'environment.json').read_text())
 if recorded['packages']!=environment()['packages']:raise ValueError('Installed core versions changed; restore the run environment.')
 from package_integrity import verify
 if verify(ROOT):raise ValueError('Package integrity check failed.')
 return c

def run_stage(run,stage,resume=False):
 import fcntl,signal,shutil
 run=Path(run).resolve();status=run/'status'/f'{stage}.json'
 if stage not in STAGES:raise ValueError(stage)
 with (run/'.stage.lock').open('a') as lock:
  try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('Another controller is already running this experiment.')
  if status.exists():
   if not resume:raise RuntimeError('Stage already attempted. Use resume with this same run directory.')
   validate_resume(run);old=json.loads(status.read_text())
   if old.get('execution_status')=='COMPLETED':
    receipt=run/'receipts'/f'{stage}.json'
    if not receipt.is_file() or json.loads(receipt.read_text())!=primary_receipt(run,stage):raise RuntimeError('Saved stage outputs changed or receipt is missing: '+stage)
    print(stage+': verified completed outputs; skipped.',flush=True);return old
   # Residential cells share fitted state. Never pretend that replaying a suffix
   # in a new interpreter recovers that state correctly.
   if stage=='residential':raise RuntimeError('Residential stage did not complete. Preserve the run and its log; automatic in-stage replay is disabled. Completed earlier stages remain saved.')
   stamp=time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
   history=run/'attempt_history'/stamp/stage;history.mkdir(parents=True)
   for path in [status,run/'logs'/f'{stage}.log']:
    if path.exists():shutil.move(str(path),history/path.name)
   dirs={'controlled':'controlled_stage','smartstar':'smartstar','toniot':'toniot','inspection':'inspection','protocol_regression':'protocol_regression','window_reconstruction':'window_reconstruction','paper_comparison':'paper_comparison','figures':'figures'}
   if stage in dirs and (run/dirs[stage]).exists():shutil.move(str(run/dirs[stage]),history/'previous_outputs')
  checks=json.loads((run/'preflight.json').read_text())
  if any(x['status']!='PASS' for x in checks):raise RuntimeError('Preflight blocked; inspect preflight.csv.')
  previous=STAGES[:STAGES.index(stage)]
  missing=[x for x in previous if not (run/'status'/f'{x}.json').is_file() or json.loads((run/'status'/f'{x}.json').read_text()).get('execution_status')!='COMPLETED']
  if missing:raise RuntimeError('Incomplete preceding stages: '+str(missing))
  log=run/'logs'/f'{stage}.log';log.parent.mkdir(exist_ok=True)
  write_json(status,{'stage':stage,'execution_status':'RUNNING','paper_agreement':'NOT_TESTED'})
  start=time.monotonic();heartbeat=start
  with log.open('w') as f:
   proc=subprocess.Popen([sys.executable,str(ROOT/'src/worker.py'),'--run',str(run),'--stage',stage],cwd=run,env=process_environment(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
   try:
    while proc.poll() is None:
     time.sleep(2)
     if time.monotonic()-heartbeat>=30:print(f'{stage}: {(time.monotonic()-start)/60:.1f} min; {log}',flush=True);heartbeat=time.monotonic()
   except BaseException:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    write_json(status,{'stage':stage,'execution_status':'INTERRUPTED'});raise
  result=json.loads(status.read_text());result.update(elapsed_seconds=round(time.monotonic()-start,2),log=str(log))
  if proc.returncode or result.get('execution_status')!='COMPLETED':
   result['execution_status']='FAILED';write_json(status,result)
   with log.open('rb') as f:f.seek(max(0,log.stat().st_size-6000));print(f.read().decode(errors='replace'))
   raise RuntimeError(stage+' failed. Preserve this run and inspect its diagnostic archive.')
  try:write_json(run/'receipts'/f'{stage}.json',primary_receipt(run,stage))
  except Exception as e:
   result.update(execution_status='FAILED',error='Completion receipt failed: '+str(e));write_json(status,result);raise
  write_json(status,result);print(stage,result['execution_status'],result.get('paper_agreement','NOT_TESTED'),flush=True)
  return result

def summary(run):
 run=Path(run);rows=[]
 for stage in STAGES:
  p=run/'status'/f'{stage}.json';rows.append(json.loads(p.read_text()) if p.is_file() else {'stage':stage,'execution_status':'NOT_RUN'})
 complete=all(x.get('execution_status')=='COMPLETED' for x in rows)
 p=run/'paper_comparison/SUMMARY.json';comparison=json.loads(p.read_text()) if p.is_file() else {'status':'NOT_RUN'}
 result={'stages':rows,'all_stages_completed':complete,'paper_reference_status':comparison['status'],'registered_result_agreement':complete and comparison['status']=='PASS_FOR_VERSIONED_REFERENCE','exhaustive_manuscript_coverage':False,'exact_repeatability_verified':False,'q4_manifest_origin':'supplied_data_derived_train_val_input','upstream_alignment_regenerated':False,'window_analysis_origin':'documented_reconstruction','meaning':'A completed fresh run and matching registered results are distinct from repeatability across two independent runs.'}
 write_json(run/'FINAL_REPORT.json',result);return result
