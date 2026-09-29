"""Collect one small diagnostic archive without source datasets or trained weights."""
from pathlib import Path
import argparse,zipfile,json

def collect(root):
 root=Path(root).resolve();dest=root.parent/(root.name+'_DIAGNOSTICS.zip');included=[];excluded=[]
 if dest.exists():
  import datetime
  dest=root.parent/(root.name+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_DIAGNOSTICS.zip')
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(root.rglob('*')):
   if not p.is_file():continue
   rel=p.relative_to(root)
   if 'recovery_history' in rel.parts or 'attempt_history' in rel.parts:continue
   keep=(p.suffix in {'.csv','.json','.jsonl'} and (len(rel.parts)==1 or rel.parts[0] in {'status','window_reconstruction','protocol_regression','figures','paper_comparison','receipts'} or 'reports' in rel.parts or 'tables' in rel.parts)) or rel.parts[0]=='logs' or p.name in {'input_inventory.json','stdout.txt','stderr.txt','worker_config.json','worker_result.json','environment.lock.txt'}
   if not keep:continue
   if p.stat().st_size>10*1024**2:
    excluded.append({'file':str(rel),'reason':'larger than 10 MiB'})
    if p.suffix=='.log' or p.name in {'stdout.txt','stderr.txt'}:
     with p.open('rb') as f:f.seek(max(0,p.stat().st_size-128000));tail=f.read()
     z.writestr(str(rel)+'.tail.txt',tail)
    continue
   z.write(p,rel);included.append(str(rel))
  z.writestr('COLLECTION.json',json.dumps({'included':included,'excluded':excluded,'source_datasets_included':False,'trained_weights_included':False},indent=2))
 return dest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);print(collect(p.parse_args().run))
