"""Validate the bundled data-derived coupling manifest without training or writing files."""
from pathlib import Path
import json, sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from manifest_validation import validate_manifest
if __name__ == '__main__':
    result = validate_manifest(ROOT)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))
