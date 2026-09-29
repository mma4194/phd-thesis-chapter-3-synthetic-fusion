"""Compare two complete fresh runs; exit 2 if differences or gaps remain."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from repeatability import compare_runs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('first', type=Path)
parser.add_argument('second', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
result = compare_runs(args.first, args.second, args.output)
print(json.dumps(result, indent=2))
sys.exit(0 if result['status'] == 'PASS_FOR_LISTED_OUTPUTS' else 2)
