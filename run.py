#!/usr/bin/env python3
"""Run or resume the same pipeline used by the master notebook."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from configuration import load_config
from workflow import new_run,preflight,run_stage,summary,validate_resume,STAGES
from collect_run import collect

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path);p.add_argument('--resume',type=Path)
 p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if a.resume and a.config:p.error('Resume reads the existing run configuration; omit --config.')
 if a.resume:
  run=a.resume.resolve();validate_resume(run)
 else:run=new_run(load_config(a.config))
 print('RUN_DIRECTORY:',run,flush=True)
 error=None
 try:
  checks=preflight(run)
  if any(x['status']!='PASS' for x in checks):raise RuntimeError(f'Preflight blocked: {run}/preflight.csv')
  if not a.preflight_only:
   for stage in STAGES:run_stage(run,stage,resume=bool(a.resume))
 except BaseException as e:error=e
 finally:
  report=summary(run);print(json.dumps(report,indent=2));print('DIAGNOSTIC_ARCHIVE:',collect(run),flush=True)
 if error:raise error
 if not a.preflight_only and not report['registered_result_agreement']:return 2
 return 0

if __name__=='__main__':sys.exit(main())
