import os
import json
from pathlib import Path
from collections import Counter
import numpy as np

if "CFG" not in globals() or not isinstance(CFG, dict):
    raise RuntimeError("Run Cell 1 first so CFG is defined.")

OUTDIR_108X = Path(str(globals().get("OUTDIR", CFG["outdir"]))).expanduser().resolve()
ARTDIR_108X = OUTDIR_108X / "artifacts"
REPDIR_108X = OUTDIR_108X / "reports"
CANDDIR_108X = OUTDIR_108X / "portfolio_candidates"
CONTRACT_DIR_108X = ARTDIR_108X / "contracts"
CONTRACT_DIR_108X.mkdir(parents=True, exist_ok=True)
REPDIR_108X.mkdir(parents=True, exist_ok=True)

selected_json_path = ARTDIR_108X / "cell10_selected_generator_by_col.json"
selected_rich_json_path = ARTDIR_108X / "cell10_selected_generator_by_col_rich.json"
selection_summary_path = ARTDIR_108X / "cell10_portfolio_selection_summary.json"
for p in [selected_json_path, selected_rich_json_path, selection_summary_path]:
    if not p.exists():
        raise RuntimeError(f"Run Cell 10.8 first; missing {p}")

selected = json.loads(selected_json_path.read_text(encoding="utf-8"))
rich = json.loads(selected_rich_json_path.read_text(encoding="utf-8"))
summary = json.loads(selection_summary_path.read_text(encoding="utf-8"))

# Predeclared stricter margins. Change these only before running Cell 10.8/10.9; never after TEST QA.
CFG.setdefault("cell10_8x_min_score_gain_frac", 0.005)   # 0.5% total score improvement
CFG.setdefault("cell10_8x_min_q1_gain_frac", 0.000)      # allow temporal-only wins if total score clears
CFG.setdefault("cell10_8x_min_q2_gain_frac", 0.000)
CFG.setdefault("cell10_8x_require_block_stability", True)

min_score_gain = float(CFG["cell10_8x_min_score_gain_frac"])
min_q1_gain = float(CFG["cell10_8x_min_q1_gain_frac"])
min_q2_gain = float(CFG["cell10_8x_min_q2_gain_frac"])
require_block = bool(CFG["cell10_8x_require_block_stability"])

a1_val_path = str(CANDDIR_108X / "A1_VAL_baseline.parquet")
a1_test_path = str(CANDDIR_108X / "A1_TEST_baseline.parquet")
A1_GEN = "A1_temporal_block_bootstrap"

def _finite_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

kept, demoted, already_a1 = [], [], []
for col, rec in list(selected.items()):
    gen = str(rec.get("selected_generator", ""))
    r = rich.get(col, {}) if isinstance(rich.get(col, {}), dict) else {}
    if gen == A1_GEN:
        already_a1.append(col)
        continue

    score_gain = _finite_float(r.get("score_gain_frac", rec.get("score_gain_frac", 0.0)))
    q1_gain = _finite_float(r.get("q1_gain_frac", rec.get("q1_gain_frac", 0.0)))
    q2_gain = _finite_float(r.get("q2_gain_frac", rec.get("q2_gain_frac", 0.0)))
    block_ok = bool(r.get("block_stability_passed", True))
    passes = (
        score_gain >= min_score_gain
        and (q1_gain >= min_q1_gain or q2_gain >= min_q2_gain)
        and ((not require_block) or block_ok)
    )
    if passes:
        kept.append({"col": col, "selected_generator": gen, "score_gain_frac": score_gain, "q1_gain_frac": q1_gain, "q2_gain_frac": q2_gain})
        continue

    old = dict(rec)
    baseline_score = _finite_float(r.get("baseline_score", r.get("selected_score", rec.get("score", 0.0))))
    baseline_q1 = _finite_float(r.get("baseline_q1_score", r.get("selected_q1_score", rec.get("q1_score", 0.0))))
    baseline_q2 = _finite_float(r.get("baseline_q2_score", r.get("selected_q2_score", rec.get("q2_score", 0.0))))
    selected[col] = {
        "selected_generator": A1_GEN,
        "selected_candidate_id": "A1_VAL_baseline",
        "selected_candidate_uid": "A1_temporal_block_bootstrap::A1_VAL_baseline",
        "reason": "cell10_8x_VAL_gate_demoted_non_A1_before_TEST",
        "val_path": a1_val_path,
        "test_path": a1_test_path,
        "requires_downstream_test_materialization": False,
        "score": baseline_score,
        "q1_score": baseline_q1,
        "q2_score": baseline_q2,
        "selection_baseline": "A1_VAL_baseline",
    }
    rich[col] = dict(selected[col])
    rich[col].update({
        "col": col,
        "tier": r.get("tier", ""),
        "baseline_score": baseline_score,
        "baseline_q1_score": baseline_q1,
        "baseline_q2_score": baseline_q2,
        "selected_score": baseline_score,
        "selected_q1_score": baseline_q1,
        "selected_q2_score": baseline_q2,
        "score_gain_frac": 0.0,
        "q1_gain_frac": 0.0,
        "q2_gain_frac": 0.0,
        "ks_gain": 0.0,
        "wasserstein_delta": 0.0,
        "support_jaccard_delta": 0.0,
        "cell10_8x_demoted_from_generator": gen,
        "cell10_8x_demoted_from_candidate_id": old.get("selected_candidate_id", ""),
        "cell10_8x_demote_reason": "failed_predeclared_VAL_margin_or_block_stability_gate",
    })
    demoted.append({
        "col": col,
        "old_generator": gen,
        "score_gain_frac": score_gain,
        "q1_gain_frac": q1_gain,
        "q2_gain_frac": q2_gain,
        "block_stability_passed": block_ok,
    })

counts = Counter(str(v.get("selected_generator", "")) for v in selected.values())
non_a1_cols = sorted([c for c, v in selected.items() if str(v.get("selected_generator", "")) != A1_GEN])
summary["selected_counts"] = dict(sorted(counts.items()))
summary["selected_non_A1_cols"] = non_a1_cols
summary["selected_non_A1_count"] = int(len(non_a1_cols))
summary["all_A1_selection_is_valid"] = bool(len(non_a1_cols) == 0)
summary["cell10_8x_val_gate"] = {
    "applied": True,
    "TEST_real_values_used": False,
    "min_score_gain_frac": min_score_gain,
    "min_q1_gain_frac": min_q1_gain,
    "min_q2_gain_frac": min_q2_gain,
    "require_block_stability": require_block,
    "already_A1_n": len(already_a1),
    "kept_non_A1_n": len(kept),
    "demoted_to_A1_n": len(demoted),
}

selected_json_path.write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
selected_rich_json_path.write_text(json.dumps(rich, indent=2, sort_keys=True), encoding="utf-8")
selection_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

import pandas as pd
pd.DataFrame(kept).to_csv(REPDIR_108X / "cell10_8x_val_gate_kept_non_A1.csv", index=False)
pd.DataFrame(demoted).to_csv(REPDIR_108X / "cell10_8x_val_gate_demoted_to_A1.csv", index=False)
contract = {
    "cell": "10.8x",
    "role": "VAL_only_protocol_A2_precommit_gate",
    "TEST_real_values_used": False,
    "selection_files_overwritten_before_TEST_materialization": True,
    "already_A1_n": len(already_a1),
    "kept_non_A1_n": len(kept),
    "demoted_to_A1_n": len(demoted),
    "final_non_A1_n": len(non_a1_cols),
}
(CONTRACT_DIR_108X / "cell10_8x_val_only_a2_precommit_gate_THESIS.json").write_text(
    json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8"
)
print("[Cell10.8x] VAL-only A2 gate complete:", contract)