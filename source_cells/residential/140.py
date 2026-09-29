import json
from pathlib import Path
import pandas as pd

required = ["CFG", "CELL15_6_PUBLIC_CPS_PATH", "CELL15_6_PUBLIC_PROTOCOL_PATH", "CELL15_6_PUBLIC_IOT_PATH"]
missing = [k for k in required if k not in globals()]
if missing:
    raise RuntimeError(f"Run Cell 15.6 first; missing globals: {missing}")

OUTDIR_BASE = Path(str(globals().get("OUTDIR", CFG["outdir"]))).expanduser().resolve()
PUBLIC_Q6_DIR = OUTDIR_BASE / "q6_public_reaudit_v2"
PUBLIC_Q6_REPORT_DIR = PUBLIC_Q6_DIR / "reports"
PUBLIC_Q6_SYN_DIR = PUBLIC_Q6_DIR / "synthetic"
PUBLIC_Q6_CONTRACT_DIR = PUBLIC_Q6_DIR / "artifacts" / "contracts"
for d in [PUBLIC_Q6_REPORT_DIR, PUBLIC_Q6_SYN_DIR, PUBLIC_Q6_CONTRACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

public_cps = pd.read_parquet(CELL15_6_PUBLIC_CPS_PATH)
public_protocol = pd.read_parquet(CELL15_6_PUBLIC_PROTOCOL_PATH)
public_iot = pd.read_parquet(CELL15_6_PUBLIC_IOT_PATH)

PUBLIC_Q6_CONTEXT = {
    "name": "public_q6_candidate",
    "outdir": str(PUBLIC_Q6_DIR),
    "report_dir": str(PUBLIC_Q6_REPORT_DIR),
    "synthetic_dir": str(PUBLIC_Q6_SYN_DIR),
    "candidate_paths": {
        "cps": str(CELL15_6_PUBLIC_CPS_PATH),
        "protocol": str(CELL15_6_PUBLIC_PROTOCOL_PATH),
        "iot": str(CELL15_6_PUBLIC_IOT_PATH),
    },
    "candidate_shapes": {
        "cps": list(public_cps.shape),
        "protocol": list(public_protocol.shape),
        "iot": list(public_iot.shape),
    },
    "scientific_cell15_0_globals_mutated": False,
    "test_values_role": "privacy_reference_only_in_dedicated_public_Q6_audit",
    "next_step": "Use dedicated public-Q6 audit cells parameterized by PUBLIC_Q6_CONTEXT; do not rerun Cell15.1-15.4 by rebinding globals.",
}
(PUBLIC_Q6_CONTRACT_DIR / "cell15_7a_public_q6_context_no_global_rebinding_THESIS.json").write_text(
    json.dumps(PUBLIC_Q6_CONTEXT, indent=2, sort_keys=True), encoding="utf-8"
)
globals()["PUBLIC_Q6_CONTEXT"] = PUBLIC_Q6_CONTEXT
print("[Cell15.7a replacement] Created PUBLIC_Q6_CONTEXT without rebinding scientific privacy globals.")
print("[Cell15.7a replacement] Public candidate shapes:", PUBLIC_Q6_CONTEXT["candidate_shapes"])