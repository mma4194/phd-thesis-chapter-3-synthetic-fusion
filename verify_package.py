from pathlib import Path
import sys,json
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'src'))
from package_integrity import verify
bad=verify(root);print(json.dumps({'status':'PASS' if not bad else 'DIFFERENT','failures':bad},indent=2));raise SystemExit(bool(bad))
