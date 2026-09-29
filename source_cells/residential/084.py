# %% CELL 12.f.V5.1 — Observability-mask V5 TRAIN→VAL backtest selector
# Purpose:
#   Use TRAIN→VAL backtesting to select materializer classes and claim scope.
#   This is allowed because every design choice is TRAIN/VAL-only.

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if "CELL12F_V5_MASK_CACHE" not in globals():
    raise RuntimeError("[12.f.V5.1] Run 12.f.V5.0 first.")


def _ks_12f5(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 1.0
    a = np.sort(a)
    b = np.sort(b)
    x = np.unique(np.r_[a, b])
    ca = np.searchsorted(a, x, side="right") / len(a)
    cb = np.searchsorted(b, x, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _wasserstein_12f5(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0:
        return float(np.mean(np.abs(b)))
    if len(b) == 0:
        return float(np.mean(np.abs(a)))
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        qs = np.linspace(0, 1, 101)
        return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _state_metrics_12f5(real, fake, state=None):
    rl, rs = _run_lengths_12f5(real)
    sl, ss = _run_lengths_12f5(fake)
    if state is not None:
        rl = rl[rs == state]
        sl = sl[ss == state]
    return {
        "ks": _ks_12f5(rl, sl),
        "wasserstein": _wasserstein_12f5(rl, sl),
        "real_max": int(rl.max()) if len(rl) else 0,
        "syn_max": int(sl.max()) if len(sl) else 0,
    }


def _window_features_12f5(vals, window=1024):
    vals = np.asarray(vals, dtype=np.uint8)
    if len(vals) == 0:
        return np.zeros((0, 3), dtype=float)
    m = max(1, len(vals) // window)
    vals = vals[:m * window]
    chunks = vals.reshape(m, window)
    rates = chunks.mean(axis=1)
    trans = np.mean(chunks[:, 1:] != chunks[:, :-1], axis=1)
    longest = []
    for row in chunks:
        lens, _ = _run_lengths_12f5(row)
        longest.append(float(lens.max() / len(row)) if len(lens) else 1.0)
    return np.c_[rates, trans, np.asarray(longest)]


def _c2st_auc_12f5(real, fake, seed=37):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        Xr = _window_features_12f5(real)
        Xs = _window_features_12f5(fake)
        n = min(len(Xr), len(Xs))
        if n < 8:
            return 0.5
        X = np.vstack([Xr[:n], Xs[:n]])
        y = np.r_[np.zeros(n), np.ones(n)]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, p))
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


def _make_candidate_12f5(kind, base_vals, n, seed):
    if kind == "exact_one_constant":
        return np.ones(n, dtype=np.uint8)
    if kind == "exact_zero_constant":
        return np.zeros(n, dtype=np.uint8)
    if kind == "tail_replay":
        return _tail_replay_12f5(base_vals, n)
    if kind == "head_replay":
        return _head_replay_12f5(base_vals, n)
    if kind == "circular_replay":
        return _circular_replay_12f5(base_vals, n, seed)
    if kind == "block_bootstrap_8192":
        return _block_bootstrap_12f5(base_vals, n, seed, block=8192)
    if kind == "block_bootstrap_32768":
        return _block_bootstrap_12f5(base_vals, n, seed, block=32768)
    if kind == "run_bootstrap":
        return _run_bootstrap_12f5(base_vals, n, seed)
    if kind == "rare_episode_bootstrap":
        return _rare_episode_bootstrap_12f5(base_vals, n, seed)
    if kind == "exact_rate_spread":
        return _exact_rate_spread_12f5(base_vals, n, seed)
    raise ValueError(kind)


def _qa_for_selector_12f5(real, fake):
    rs = _stats_12f5(real)
    fs = _stats_12f5(fake)
    all_m = _state_metrics_12f5(real, fake, state=None)
    c2st_auc = _c2st_auc_12f5(real, fake, seed=11)

    rate_error = abs(rs["state_one_rate"] - fs["state_one_rate"])
    transition_error = abs(rs["transition_rate"] - fs["transition_rate"])

    reasons = []
    if rate_error > 0.05:
        reasons.append("state_rate_error_gt_0.05")
    elif rate_error > 0.02:
        reasons.append("state_rate_error_gt_0.02")
    if all_m["ks"] > 0.50:
        reasons.append("runlength_ks_gt_0.50")
    elif all_m["ks"] > 0.25:
        reasons.append("runlength_ks_gt_0.25")
    if all_m["wasserstein"] > 50000:
        reasons.append("dwell_wasserstein_gt_50000")
    elif all_m["wasserstein"] > 10000:
        reasons.append("dwell_wasserstein_gt_10000")
    elif all_m["wasserstein"] > 1000:
        reasons.append("dwell_wasserstein_gt_1000")
    if c2st_auc > 0.75:
        reasons.append("mask_c2st_auc_gt_0.75")
    elif c2st_auc > 0.65:
        reasons.append("mask_c2st_auc_gt_0.65")

    if not reasons:
        status = "pass"
        reasons = ["within_thresholds"]
    elif any(x in reasons for x in ["state_rate_error_gt_0.05", "runlength_ks_gt_0.50", "dwell_wasserstein_gt_50000", "mask_c2st_auc_gt_0.75"]):
        status = "fatal"
    else:
        status = "warning"

    score = (
        4.0 * min(rate_error / 0.05, 4.0)
        + 1.5 * min(all_m["ks"] / 0.50, 4.0)
        + 1.0 * min(all_m["wasserstein"] / 50000.0, 4.0)
        + 1.0 * min(max(0.0, c2st_auc - 0.5) / 0.25, 4.0)
        + 0.5 * min(transition_error / 0.01, 4.0)
    )
    if status == "fatal":
        score += 100.0
    elif status == "warning":
        score += 10.0

    return {
        "status": status,
        "reasons": ";".join(reasons),
        "score": float(score),
        "state_one_rate_error": float(rate_error),
        "transition_error": float(transition_error),
        "runlength_ks": float(all_m["ks"]),
        "dwell_wasserstein": float(all_m["wasserstein"]),
        "mask_c2st_auc": float(c2st_auc),
        "real_state_one_rate": float(rs["state_one_rate"]),
        "synthetic_state_one_rate": float(fs["state_one_rate"]),
    }


candidate_kinds_default = [
    "tail_replay", "head_replay", "circular_replay",
    "block_bootstrap_8192", "block_bootstrap_32768",
    "run_bootstrap", "rare_episode_bootstrap", "exact_rate_spread"
]

selector_rows, backtest_rows = [], []

for col in CELL12F_V5_TARGETS:
    cache = CELL12F_V5_MASK_CACHE[col]
    tr, va, tv = cache["tr"], cache["va"], cache["tv"]
    tr_s, va_s, tv_s = cache["tr_stats"], cache["va_stats"], cache["tv_stats"]

    tr_rate, va_rate = tr_s["state_one_rate"], va_s["state_one_rate"]
    rate_delta = abs(tr_rate - va_rate) if np.isfinite(tr_rate) and np.isfinite(va_rate) else np.nan
    transition_delta = abs(tr_s["transition_rate"] - va_s["transition_rate"]) if np.isfinite(tr_s["transition_rate"]) and np.isfinite(va_s["transition_rate"]) else np.nan

    name = col.lower()
    is_entity_indicator = ("__entity_obs__" in name) or ("__entity_stale__" in name)
    is_core_global = col in {"iot__tier_present"}

    same_constant = tr_s["support"] in {"constant_0", "constant_1"} and tr_s["support"] == va_s["support"]
    train_or_val_empty = tr_s["support"] == "empty" or va_s["support"] == "empty"
    support_shift = (
        (tr_s["support"] in {"constant_0", "constant_1"} and va_s["support"] not in {tr_s["support"], "empty"})
        or (va_s["support"] in {"constant_0", "constant_1"} and tr_s["support"] not in {va_s["support"], "empty"})
    )
    opposite_rare_constant = (
        (tr_s["support"] in {"rare_1", "ultra_rare_1"} and va_s["support"] == "constant_0")
        or (va_s["support"] in {"rare_1", "ultra_rare_1"} and tr_s["support"] == "constant_0")
        or (tr_s["support"] in {"rare_0", "ultra_rare_0"} and va_s["support"] == "constant_1")
        or (va_s["support"] in {"rare_0", "ultra_rare_0"} and tr_s["support"] == "constant_1")
    )
    min_edge_mass = max(1e-9, min(
        tr_rate if np.isfinite(tr_rate) else 0.0,
        va_rate if np.isfinite(va_rate) else 0.0,
        1.0 - tr_rate if np.isfinite(tr_rate) else 0.0,
        1.0 - va_rate if np.isfinite(va_rate) else 0.0,
    ))
    rate_unstable = bool(np.isfinite(rate_delta) and rate_delta > max(0.015, 0.40 * min_edge_mass))
    transition_unstable = bool(
        np.isfinite(transition_delta)
        and transition_delta > max(0.0015, 2.5 * max(tr_s["transition_rate"], va_s["transition_rate"], 1e-9))
    )

    pre_scope_excluded = False
    pre_scope_reason = ""
    if train_or_val_empty:
        pre_scope_excluded = True
        pre_scope_reason = "TRAIN or VAL support is empty."
    elif same_constant and is_core_global:
        pre_scope_excluded = False
    elif same_constant and is_entity_indicator:
        pre_scope_excluded = True
        pre_scope_reason = "Entity indicator exact TRAIN/VAL constant is brittle; matrix-completion only."
    elif same_constant:
        pre_scope_excluded = True
        pre_scope_reason = "Non-core exact TRAIN/VAL constant is non-identifiable for temporal claim."
    elif support_shift or opposite_rare_constant:
        pre_scope_excluded = True
        pre_scope_reason = "TRAIN/VAL support shift or rare-to-opposite-constant shift."
    elif rate_unstable:
        pre_scope_excluded = True
        pre_scope_reason = "TRAIN/VAL state rate drift exceeds strict bound."
    elif transition_unstable:
        pre_scope_excluded = True
        pre_scope_reason = "TRAIN/VAL transition drift exceeds strict bound."

    candidate_kinds = list(candidate_kinds_default)
    if tr_s["support"] == "constant_1":
        candidate_kinds.append("exact_one_constant")
    if tr_s["support"] == "constant_0":
        candidate_kinds.append("exact_zero_constant")

    bts = []
    for kind in candidate_kinds:
        try:
            pred_va = _make_candidate_12f5(kind, tr, len(va), _seed_12f5(col + "::" + kind + "::v5_val"))
            qa = _qa_for_selector_12f5(va, pred_va)
            row = {
                "col": col, "candidate_materializer_v5": kind,
                "val_backtest_status": qa["status"], "val_backtest_reasons": qa["reasons"],
                "val_backtest_score": qa["score"],
                "val_state_one_rate_error": qa["state_one_rate_error"],
                "val_transition_error": qa["transition_error"],
                "val_runlength_ks": qa["runlength_ks"],
                "val_dwell_wasserstein": qa["dwell_wasserstein"],
                "val_mask_c2st_auc": qa["mask_c2st_auc"],
                "TEST_real_values_used": False,
            }
        except Exception as e:
            row = {
                "col": col, "candidate_materializer_v5": kind,
                "val_backtest_status": "fatal",
                "val_backtest_reasons": f"candidate_generation_error:{e}",
                "val_backtest_score": 9999.0,
                "val_state_one_rate_error": np.nan,
                "val_transition_error": np.nan,
                "val_runlength_ks": np.nan,
                "val_dwell_wasserstein": np.nan,
                "val_mask_c2st_auc": np.nan,
                "TEST_real_values_used": False,
            }
        bts.append(row)
        backtest_rows.append(row)

    bt_df = pd.DataFrame(bts)
    best = bt_df.sort_values(["val_backtest_score", "candidate_materializer_v5"], ascending=[True, True]).iloc[0].to_dict()

    if pre_scope_excluded:
        claim_status = "scope_excluded"
        selected_generator = "MaskV5ScopeExcludedTrainValInstability"
        materializer = "scope_excluded_tail_replay"
        reason = pre_scope_reason
        included, excluded = False, True
    elif same_constant and is_core_global:
        claim_status = "included_pass"
        selected_generator = "MaskV5CoreExactConstant"
        materializer = "exact_one_constant" if tr_s["support"] == "constant_1" else "exact_zero_constant"
        reason = "Core global exact TRAIN/VAL constant."
        included, excluded = True, False
    elif best["val_backtest_status"] == "pass":
        claim_status = "included_pass"
        selected_generator = f"MaskV5BacktestSelected_{best['candidate_materializer_v5']}"
        materializer = best["candidate_materializer_v5"]
        reason = "Selected by TRAIN→VAL backtest with pass status."
        included, excluded = True, False
    elif best["val_backtest_status"] == "warning":
        claim_status = "included_warning"
        selected_generator = f"MaskV5BacktestSelected_{best['candidate_materializer_v5']}"
        materializer = best["candidate_materializer_v5"]
        reason = "Selected by TRAIN→VAL backtest with warning status."
        included, excluded = True, False
    else:
        claim_status = "scope_excluded"
        selected_generator = "MaskV5ScopeExcludedNoValSafeMaterializer"
        materializer = "scope_excluded_tail_replay"
        reason = "No candidate materializer passed/warned TRAIN→VAL backtest."
        included, excluded = False, True

    selector_rows.append({
        "col": col, "semantic_mode_v5": cache["mode"],
        "selected_generator": selected_generator, "materializer_class_v5": materializer,
        "v5_claim_status_trainval_only": claim_status,
        "claim_included": included, "scope_excluded": excluded,
        "selection_reason_trainval_only": reason,
        "train_support": tr_s["support"], "val_support": va_s["support"],
        "train_state_one_rate": tr_rate, "val_state_one_rate": va_rate,
        "trainval_state_one_rate": tv_s["state_one_rate"],
        "train_transition_rate": tr_s["transition_rate"],
        "val_transition_rate": va_s["transition_rate"],
        "train_val_rate_delta": rate_delta,
        "train_val_transition_delta": transition_delta,
        "rate_unstable_trainval_only": rate_unstable,
        "transition_unstable_trainval_only": transition_unstable,
        "support_shift_trainval_only": support_shift,
        "opposite_rare_constant_shift_trainval_only": opposite_rare_constant,
        "best_val_backtest_materializer": best["candidate_materializer_v5"],
        "best_val_backtest_status": best["val_backtest_status"],
        "best_val_backtest_score": best["val_backtest_score"],
        "best_val_backtest_reasons": best["val_backtest_reasons"],
        "best_val_state_one_rate_error": best["val_state_one_rate_error"],
        "best_val_runlength_ks": best["val_runlength_ks"],
        "best_val_dwell_wasserstein": best["val_dwell_wasserstein"],
        "best_val_mask_c2st_auc": best["val_mask_c2st_auc"],
        "TEST_real_values_used": False,
    })

selection = pd.DataFrame(selector_rows)
backtest = pd.DataFrame(backtest_rows)

selection.to_csv(REPORT_DIR_P / "cell12f_v5_locked_mask_selection.csv", index=False)
backtest.to_csv(REPORT_DIR_P / "cell12f_v5_train_to_val_materializer_backtest.csv", index=False)
selection.to_csv(REPORT_DIR_P / "cell12f_v5_trainval_policy_precommit.csv", index=False)
selection[[
    "col", "semantic_mode_v5", "selected_generator", "materializer_class_v5",
    "v5_claim_status_trainval_only", "claim_included", "scope_excluded",
    "selection_reason_trainval_only", "best_val_backtest_materializer",
    "best_val_backtest_status", "best_val_backtest_reasons",
]].to_csv(REPORT_DIR_P / "cell12f_v5_trainval_claim_scope_registry.csv", index=False)

summary = {
    "cell": "12.f.V5.1",
    "role": "observability_mask_v5_train_to_val_backtest_selector",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "targets": int(len(selection)),
    "included_n_trainval_only": int(selection["claim_included"].sum()),
    "scope_excluded_n_trainval_only": int(selection["scope_excluded"].sum()),
    "selected_generator_counts": selection["selected_generator"].value_counts().to_dict(),
    "materializer_class_counts": selection["materializer_class_v5"].value_counts().to_dict(),
    "claim_status_counts_trainval_only": selection["v5_claim_status_trainval_only"].value_counts().to_dict(),
    "val_backtest_status_counts": selection["best_val_backtest_status"].value_counts().to_dict(),
    "TEST_real_values_used": False,
    "selection_done_here": True,
    "synthetic_values_mutated": False,
    "post_TEST_repair_done_here": False,
    "same_df_te_v5_status": "development_only_unless_fresh_holdout_used",
}
(REPORT_DIR_P / "cell12f_v5_selection_policy_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
(CONTRACT_DIR_P / "cell12f_v5_selection_policy_contract.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

CELL12F_V5_SELECTION = selection
CELL12F_V5_BACKTEST = backtest
CELL12F_V5_SELECTION_SUMMARY = summary

_f5_log(f"V5 selection policy complete | targets={len(selection)} | included={summary['included_n_trainval_only']} | scope_excluded={summary['scope_excluded_n_trainval_only']}")
_f5_log(f"Contract: {CONTRACT_DIR_P / 'cell12f_v5_selection_policy_contract.json'}")

if display is not None:
    display(selection["selected_generator"].value_counts().rename_axis("selected_generator").reset_index(name="n"))
    display(selection["v5_claim_status_trainval_only"].value_counts().rename_axis("claim_status").reset_index(name="n"))
    display(selection["best_val_backtest_status"].value_counts().rename_axis("best_val_backtest_status").reset_index(name="n"))

