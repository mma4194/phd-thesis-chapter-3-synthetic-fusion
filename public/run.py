#!/usr/bin/env python3
"""Single entry point. Missing inputs and mismatches never become a PASS."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def save(p,obj):p.write_text(json.dumps(obj,indent=2,default=str)+'\n',encoding='utf-8')

def inventory(root,files):
    return [{'relative_path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(files)]

def dependencies():
    result={}
    for n in ['numpy','pandas','scipy','scikit-learn','matplotlib','pyarrow']:
        result[n]=importlib.metadata.version(n)
    return result

def compare_csv(*args,**kwargs):
    """Record numerical differences without mistaking them for execution failures."""
    from verify import compare_csv as strict_compare_csv
    try:return {'status':'PASS',**strict_compare_csv(*args,**kwargs)}
    except ValueError as e:return {'status':'DIFFERENT','reason':str(e)}

def comparison_differs(value):
    if isinstance(value,dict):
        return value.get('status')=='DIFFERENT' or any(comparison_differs(x) for x in value.values())
    return False

def check_integrity():
    rows=json.loads((ROOT/'MANIFEST.json').read_text())['files'];bad=[]
    for r in rows:
        p=ROOT/r['path']
        if not p.is_file() or digest(p)!=r['sha256']:bad.append(r['path'])
    if bad:raise ValueError('Package integrity mismatch: '+', '.join(bad))
    return {'status':'PASS','files':len(rows),'meaning':'Archive integrity only, not scientific validation'}

def job(script,out,env,timeout):
    t=time.perf_counter()
    try:
        p=subprocess.run([sys.executable,str(ROOT/'src'/script)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=timeout or None)
        txt=p.stdout+'\n'+p.stderr
    except subprocess.TimeoutExpired as e:
        txt='TIMEOUT: partial output\n'+str(e.stdout or '')+'\n'+str(e.stderr or '')
        for k in ['SMARTSTAR_ROOT','SMARTSTAR_OUT','TONIOT_ROOT','TONIOT_OUT_ROOT']:
            if env.get(k):txt=txt.replace(env[k],'<'+k.lower()+'>')
        (out/(script+'.log')).write_text(txt.replace(str(ROOT),'<artifact>'),encoding='utf-8')
        raise TimeoutError('Run exceeded timeout; no PASS claimed. Use a new output directory for retry.')
    for k in ['SMARTSTAR_ROOT','SMARTSTAR_OUT','TONIOT_ROOT','TONIOT_OUT_ROOT']:
        if env.get(k):txt=txt.replace(env[k],'<'+k.lower()+'>')
    (out/(script+'.log')).write_text(txt.replace(str(ROOT),'<artifact>'),encoding='utf-8')
    if p.returncode:raise RuntimeError(f'{script} failed (exit {p.returncode}); see its log')
    return round(time.perf_counter()-t,3)

def toniot_compare(out):
    ref=ROOT/'reference/toniot';p=out/'physical_q4'
    results={'role_manifest':compare_csv(out/'manifests/toniot_role_owner_manifest.csv',ref/'toniot_role_owner_manifest.csv',['column'])}
    for folder,n,keys,cols in [('manifests','toniot_physical_q4_manifest_trainval_frozen.csv',['pair_id'],None),('ledgers','toniot_physical_q4_test_event_response_audit.csv',['pair_id'],None),('tables','toniot_q6_release_scope_table_physical_q4.csv',['column'],['owner_role','q6_keep','q6_release_status'])]:
        results[n]=compare_csv(p/folder/n,ref/n,keys,cols,rtol=1e-7,atol=1e-9)
    return results

def smartstar_compare(out):
    ref=ROOT/'reference/smartstar';result={}
    for sub,n,keys in [('ledgers','test_evidence_q1_q2_q3_ledger.csv',['feature']),('ledgers','candidate_selection_val_only_ledger.csv',['feature']),('manifests','q6_public_source_release_manifest.csv',['feature']),('manifests','split_manifest.csv',['split'])]:
        result[n]=compare_csv(out/sub/n,ref/n,keys,rtol=1e-7,atol=1e-9)
    result['C2ST']=compare_csv(out/'tables/branch_c2st_smoke_diagnostic.csv',ref/'branch_c2st_smoke_diagnostic.csv',['owner_role'],rtol=0,atol=0.0000005)
    return result

def run_public(args,out):
    import pandas as pd
    env=os.environ.copy();env.update({'MPLBACKEND':'Agg','PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    for k in list(env):
        if k.startswith(('TONIOT_','SMARTSTAR_')):env.pop(k)
    result={'status':'RUNNING','command':args.command,'full_paper_reproduced':False}
    try:
        result['environment']=dependencies();save(out/'status.json',result)
        if args.command=='smartstar':
            if args.input is None:raise FileNotFoundError('Use --input with the directory containing the six homeA-* folders. See ../docs/DATA.md.')
            inp=Path(args.input).expanduser().resolve();folders=['circuit','environmental','switch','furnace','door','motion']
            absent=[f'homeA-{x}' for x in folders if not (inp/f'homeA-{x}').is_dir()]
            if absent:raise FileNotFoundError('Missing Smart* folders: '+', '.join(absent))
            files=[p for x in folders for p in (inp/f'homeA-{x}').rglob('*') if p.is_file()]
            save(out/'input_inventory.json',inventory(inp,files))
            env.update(SMARTSTAR_ROOT=str(inp),SMARTSTAR_OUT=str(out))
            result['seconds']=job('smartstar.py',out,env,args.timeout)
            result['comparison']=smartstar_compare(out)
            result['scope']='Smart* raw-source preprocessing, synthesis, metrics and candidate selection'
        else:
            env.update(TONIOT_OUT_ROOT=str(out),TONIOT_WINDOW_SECONDS='60',TONIOT_MAX_Q4_PAIRS='50',TONIOT_Q4_LAG_WINDOWS='5',TONIOT_MIN_EVENTS_PER_SPLIT='8',TONIOT_MIN_TOTAL_EVENTS_TRAINVAL='20',TONIOT_MAX_ROWS_PER_FILE='0',TONIOT_HASH_SELECTED_SOURCES='1',TONIOT_PHYS_MIN_EVENTS_PER_SPLIT='5',TONIOT_PHYS_MIN_TOTAL_EVENTS_TRAINVAL='15',TONIOT_PHYS_OVERWRITE_CANONICAL='0')
            if args.command=='toniot':
                if args.input is None:raise FileNotFoundError('Use --input with the directory containing Processed_datasets. See ../docs/DATA.md.')
                inp=Path(args.input).expanduser().resolve();net=inp/'Processed_datasets/Processed_Network_dataset';iot=inp/'Processed_datasets/Processed_IoT_dataset'
                expected=[net/f'Network_dataset_{i}.csv' for i in range(1,24)]+[iot/f'IoT_{s}.csv' for s in ['Fridge','GPS_Tracker','Garage_Door','Modbus','Motion_Light','Thermostat','Weather']]
                absent=[p.name for p in expected if not p.is_file()]
                if absent:raise FileNotFoundError('Missing processed source files: '+', '.join(absent))
                actual=set(net.rglob('*.csv'))|set(iot.rglob('*.csv'))
                if actual!=set(expected):raise ValueError('Unexpected processed CSV files; use a dedicated copy containing only the documented 30 files.')
                save(out/'input_inventory.json',inventory(inp,expected))
                env['TONIOT_ROOT']=str(inp)
                result['base_seconds']=job('toniot_base.py',out,env,args.timeout)
                aligned_path=out/'intermediate/toniot_aligned_windows_with_split.parquet'
                pd.read_parquet(aligned_path).to_csv(out/'aligned_comparison.csv',index=False)
                result['alignment_comparison']=compare_csv(out/'aligned_comparison.csv',ROOT/'reference/toniot/toniot_aligned_windows_reference.csv.gz',['window_start'],rtol=1e-7,atol=1e-9)
                (out/'aligned_comparison.csv').unlink()
                result['scope']='TON-IoT raw processed CSVs to aligned windows, physical Q4 and Q6; real-data transfer only'
            else:
                raise ValueError('Only source-dataset execution is enabled in this package.')
            result['physical_seconds']=job('toniot_physical.py',out,env,args.timeout)
            result['comparison']=toniot_compare(out)
        result['status']='DIFFERENT' if comparison_differs(result) else 'PASS'
        result['execution_status']='COMPLETED'
    except FileNotFoundError as e:result.update(status='MISSING_INPUT',reason=str(e))
    except TimeoutError as e:result.update(status='TIMEOUT',reason=str(e))
    except Exception as e:result.update(status='FAIL',reason=str(e))
    save(out/'status.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['integrity','smartstar','toniot'])
    p.add_argument('--input',help='Public raw-data root; never the residential dataset')
    p.add_argument('--output',help='New output directory (must not already exist)')
    p.add_argument('--timeout',type=int,default=7200,help='Seconds per public-data stage; default 7200; 0 disables the limit')
    args=p.parse_args()
    if args.command=='integrity':result=check_integrity();print(json.dumps(result,indent=2));return 0
    if args.timeout<0:p.error('--timeout must be nonnegative')
    out=Path(args.output).expanduser().resolve() if args.output else ROOT/'runs'/(args.command+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%f'))
    if out.exists():p.error('Output already exists. Choose a new --output directory; existing runs are never overwritten.')
    out.mkdir(parents=True);result=run_public(args,out)
    print(json.dumps(result,indent=2))
    print('Results:',out)
    return 0 if result['status']=='PASS' else 2
if __name__=='__main__':sys.exit(main())
