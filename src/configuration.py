"""One configuration shared by the notebook and terminal runner."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1]

def default_config():
 home=Path.home()
 base=os.environ.get('SF_SCRATCH_ROOT')
 if base:scratch=Path(base).expanduser()/home.name/'Synthetic_Fusion_Chapter3'
 elif Path('/scratch/SCWF00162').is_dir():scratch=Path('/scratch/SCWF00162')/home.name/'Synthetic_Fusion_Chapter3'
 else:scratch=ROOT.parent/'synthetic-fusion-output'
 filename='cps_unified_1s_FULLGRID_FINAL_WITH_IOT_STATEFUL_STRICTMASKED_TRIMMED.parquet'
 bundled=ROOT/'datasets/residential'/filename
 return {'residential_parquet':str(bundled if bundled.is_file() else home/filename),
         'smartstar_root':str(home/'Smart_Star_Dataset/homeA'),'toniot_root':str(home/'TON_IoT'),
         'output_parent':str(scratch/'runs'),'n_jobs':1,'public_timeout_seconds':0}

def load_config(path=None):
 c=default_config()
 if path is None and (ROOT/'config/local.json').is_file():path=ROOT/'config/local.json'
 if path:
  c.update(json.loads(Path(path).read_text()))
 return c
