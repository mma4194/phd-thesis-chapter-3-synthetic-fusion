"""Verify distributed sources while allowing notebook execution outputs to change."""
from pathlib import Path
import json,hashlib

def verify(root):
 root=Path(root);bad=[]
 for row in json.loads((root/'MANIFEST.json').read_text())['files']:
  p=root/row['path']
  if not p.is_file():bad.append({'path':row['path'],'reason':'missing'});continue
  if row.get('notebook_source_sha256'):
   from workflow import notebook_source_sha256
   equal=notebook_source_sha256(p)==row['notebook_source_sha256']
  else:equal=hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
  if not equal:bad.append({'path':row['path'],'reason':'source changed'})
 return bad
