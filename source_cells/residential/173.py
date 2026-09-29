# %% CELL EXT.0.a — Environment, paths, fail-closed contract (v3)
import os, sys, json, hashlib, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd

RNG_SEED = 20260702; np.random.seed(RNG_SEED)

# ============================================================
# BIND TO ONE COMPLETE RUN. Set this to the run whose artifacts back the paper.
# From your scan, 224537 and 124806 are both complete (54 files each). Until you
# confirm which one reproduces Table 13's 0.680824, either works for a DRY RUN of
# the extension; pick the one you intend to submit before trusting the numbers.
CANONICAL_RUN = os.environ.get("CPS_CANONICAL_RUN", "").strip() or str(globals().get("OUTDIR", CFG.get("outdir", "")))
# ============================================================
ARTIFACT_ROOT = Path(CANONICAL_RUN).expanduser().resolve()
assert ARTIFACT_ROOT.exists(), f"run dir not found: {ARTIFACT_ROOT}"
assert (ARTIFACT_ROOT/"synthetic").exists(), f"no synthetic/ under {ARTIFACT_ROOT}"
assert "QUARANTINE" not in str(ARTIFACT_ROOT).upper(), "refusing quarantined run"

# Real splits live in the splits-only session dir (115408); copy/point to them.
# If they are not inside CANONICAL_RUN/synthetic, set REAL_SPLIT_DIR explicitly:
REAL_SPLIT_DIR = ARTIFACT_ROOT/"synthetic"
if not (REAL_SPLIT_DIR/"REAL_TEST_SPLIT.parquet").exists():
    _real_split_env = os.environ.get("CPS_REAL_SPLIT_DIR", "").strip()
    if not _real_split_env:
        raise RuntimeError("[EXT.0] REAL_TEST_SPLIT.parquet not found under CANONICAL_RUN/synthetic; set CPS_REAL_SPLIT_DIR to the synthetic/ directory containing REAL_*_SPLIT.parquet.")
    REAL_SPLIT_DIR = Path(_real_split_env).expanduser().resolve()

EXT_OUT = ARTIFACT_ROOT/"EXT_OUT"
for s in ("","figures","tables","manifests"): (EXT_OUT/s).mkdir(parents=True,exist_ok=True)

EXPECTED_TEST_ROWS, EXPECTED_VAL_ROWS, EXPECTED_TRAIN_ROWS = 255_540, 255_538, 766_616
EXPECTED_PUBLIC_LOGICAL_COLS = int(CFG.get("cell15_6_expected_public_cols", CFG.get("expected_public_logical_cols", 46)))

def sha256_file(p, chunk=1<<20):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(chunk),b""): h.update(b)
    return h.hexdigest()

def find_candidates(patterns, root=None, max_show=20):
    root=root or ARTIFACT_ROOT; hits=[]
    for pat in patterns:
        hits += [Path(p) for p in glob.glob(str(root/"**"/pat), recursive=True)]
    hits=[h for h in set(hits) if "QUARANTINE" not in str(h).upper()]
    return sorted(hits, key=lambda p:p.stat().st_mtime, reverse=True)[:max_show]

def _pin(rel, root=ARTIFACT_ROOT):
    if not rel: return ""
    p=Path(rel); return str(p if p.is_absolute() else (root/rel))

def resolve_one(name, patterns, required=True):
    pinned=EXT_PATHS.get(name)
    if pinned:
        p=Path(pinned); assert p.exists(), f"[EXT.0] pinned {name} missing: {p}"; return p
    cands=find_candidates(patterns)
    if len(cands)==1:
        print(f"[EXT.0] {name}: auto -> {cands[0].relative_to(ARTIFACT_ROOT)}"); return cands[0]
    msg=(f"[EXT.0] {name}: {len(cands)} candidates in {ARTIFACT_ROOT.name} for {patterns}\n  "
         + "\n  ".join(str(c) for c in cands) + "\n[ACTION] pin it in _EXT_PATHS_REL.")
    if required: raise FileNotFoundError(msg)
    print("[EXT.0][optional] "+msg); return None

_EXT_PATHS_REL = {
    "real_train":   _pin("REAL_TRAIN_SPLIT.parquet", REAL_SPLIT_DIR),
    "real_val":     _pin("REAL_VAL_SPLIT.parquet",   REAL_SPLIT_DIR),
    "real_test":    _pin("REAL_TEST_SPLIT.parquet",  REAL_SPLIT_DIR),
    "sci_artifact": "synthetic/CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet",   # FIXED: FINAL not SCIENTIFIC
    "pub_artifact": str(globals().get("CELL15_6_PUBLIC_CPS_PATH", "synthetic/CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet")),
    "role_manifest":         "artifacts/protocol_generator_role_manifest.json",
    "role_manifest_privacy": "artifacts/cell15_0_privacy_role_manifest.json",
    "cont_qa_ledger": "", "binary_qa_ledger": "", "mask_qa_ledger": "",
    "q4_a0_terminal": "", "q4_a0_frozen_policy": "", "q4_pair_profiles": "",
    "ctgan_baseline": "", "tabddpm_baseline": "", "timegan_baseline": "",
    "baseline_scope_manifest": "",
}
# real_* are already absolute; pin the rest against ARTIFACT_ROOT
EXT_PATHS = {k:(v if str(v).startswith("/") else _pin(v)) for k,v in _EXT_PATHS_REL.items()}

PATTERNS = {
    "real_test":["REAL_TEST_SPLIT.parquet"], "real_train":["REAL_TRAIN_SPLIT.parquet"],
    "real_val":["REAL_VAL_SPLIT.parquet"],
    "sci_artifact":["CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet"],
    "pub_artifact":["CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet"],
    "role_manifest":["protocol_generator_role_manifest.json"],
    "role_manifest_privacy":["cell15_0_privacy_role_manifest.json"],
    "cont_qa_ledger":["*12c6*terminal*qa*metrics*.csv","cell12c6_iot_value_final_test_qa_metrics.csv","*12c6*qa*metrics*.csv"],
    "binary_qa_ledger":["cell12d_v6_included_claim_terminal_test_qa_metrics.csv","cell12d_v6_terminal_test_qa_metrics.csv"],
    "mask_qa_ledger":["*cell12f*mask*qa*metrics*.csv","*cell12f5*qa*metrics*.csv"],
    "q4_a0_terminal":["cell14_6R1_A0_frozen_policy_terminal_scope_status.csv"],
    "q4_a0_frozen_policy":["cell14_6R0_A0_trainval_policy_summary.json"],
    "q4_pair_profiles":["cell14_6_A0_profiles.npz","cell14_0_zigbee_coupling_profiles.npz"],
    "ctgan_baseline":["ctgan_*_synthetic_test.parquet"],
    "tabddpm_baseline":["tabddpm_*_synthetic_test.parquet"],
    "timegan_baseline":["timegan_*_synthetic_test.parquet"],
    "baseline_scope_manifest":["cell17_*_baseline_manifest.json"],
}

print(f"[EXT.0] ARTIFACT_ROOT -> {ARTIFACT_ROOT.name}")
print(f"[EXT.0] REAL_SPLIT_DIR-> {REAL_SPLIT_DIR}")
REQ_A = ["real_train","real_val","real_test","sci_artifact","pub_artifact","role_manifest"]
blocking=[]
print("\n[EXT.0] resolution status:")
for k in _EXT_PATHS_REL:
    p=EXT_PATHS.get(k,"")
    if p and Path(p).exists(): st=f"PINNED {Path(p).name}"
    elif p: st=f"PIN-MISSING {p}"; blocking += [k] if k in REQ_A else []
    else:
        c=find_candidates(PATTERNS.get(k,[]))
        if len(c)==1: st=f"auto-1 {c[0].name}"
        elif not c: st="auto-0 (none)"; blocking += [k] if k in REQ_A else []
        else: st=f"auto-{len(c)} AMBIG"; blocking += [k] if k in REQ_A else []
    print(f"   {k:24s} {st}")
print(f"\n[EXT.0] {'BLOCKING: '+str(blocking) if blocking else 'all Experiment-A artifacts resolve. proceed to EXT.0.b then EXT.A.1'}")