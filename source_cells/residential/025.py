# ==========================================================
# CELL 9 — Tier masks + drivers — v11-THESIS
# (LEAKAGE-SAFE JOINT MASK BOOTSTRAP FOR VAL + TEST HORIZONS
#  + REGIME-AWARE EPISODES
#  + TARGET-AWARE RUN CALIBRATION
#  + A1/A2-COMPATIBLE SYNTHETIC MASK ARTIFACTS
#  + CLEAN MODEL/EVAL ARTIFACT SEPARATION)
#
# Purpose:
# - Generate synthetic protocol observability masks for BOTH:
#     1) VAL horizon  -> used by downstream VAL-only candidate selection/baselines
#     2) TEST horizon -> used for final synthetic outputs after all TRAIN/VAL decisions are frozen
# - Preserve joint cross-tier temporal structure using joint-state episode bootstrap.
# - Use TRAIN-only mask fitting and target calibration under the v5-THESIS policy contract.
# - Keep TEST-derived real-mask diagnostics out of fitting/model artifacts.
# - Export driver matrices without silent missing-column fabrication.
#
# Scientific contract:
# - TRAIN is used for mask-model fitting under the canonical v5-THESIS policy.
# - TEST is NEVER used for mask fitting, target calibration, model selection, or generation.
# - TEST real masks/drivers are exported only as EVAL-only globals/diagnostics.
# - tier_mask_model.json contains only TRAIN-derived fitting/target-calibration metadata plus synthetic VAL/TEST generation metadata.
# - tier_mask_eval_diagnostics.json contains separate VAL/TEST comparison diagnostics.
#
# Outputs expected downstream:
# - SYN_MASKS_VAL_FULL,  SYN_MASKS_VAL_DDPM
# - SYN_MASKS_TEST_FULL, SYN_MASKS_TEST_DDPM
# - SYN_MASKS_FULL,      SYN_MASKS_DDPM          # backward-compatible TEST aliases
# - REAL_VAL_MASKS_FULL, REAL_VAL_MASKS_DDPM
# - REAL_TEST_MASKS_FULL, REAL_TEST_MASKS_DDPM
# - drv_tr, drv_va, drv_te_real_eval
# ==========================================================

log("--- START: Cell 9 — tier masks + drivers (v11-THESIS VAL+TEST-horizon mask synthesis) ---")

import os
import json
import hashlib
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 0) Prerequisites
# ------------------------------------------------------------
need = [
    "CFG", "log",
    "df_tr", "df_va", "df_te",
    "DDPM_TIERS_USED",
    "reg_tr", "reg_va", "reg_te",
    "IOT_DRIVER_COLS",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell9] Missing prerequisites: {missing}. Run Cells 3–8.5 first.")

OUTDIR = str(CFG["outdir"])
OUT_ART = os.path.join(OUTDIR, "artifacts")
OUT_REP = os.path.join(OUTDIR, "reports")
os.makedirs(OUT_ART, exist_ok=True)
os.makedirs(OUT_REP, exist_ok=True)
OUT_CONTRACT = os.path.join(OUT_ART, "contracts")
os.makedirs(OUT_CONTRACT, exist_ok=True)

seed = int(CFG.get("seed", 1337))
rng_fit = np.random.default_rng(seed + 900)
rng_val = np.random.default_rng(seed + 901)
rng_test = np.random.default_rng(seed + 902)

# ------------------------------------------------------------
# 0b) Clean-run leakage guard inherited from Cell 1
# ------------------------------------------------------------
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(f"[Cell9] Clean mask generation forbids CFG[{_flag!r}]=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell9] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )


# ------------------------------------------------------------
# 1) Artifact / contract checks
# ------------------------------------------------------------
ddpm_tiers_path = os.path.join(OUT_ART, "ddpm_tiers_used.json")
if not os.path.exists(ddpm_tiers_path):
    raise RuntimeError(f"[Cell9] Missing locked tier artifact: {ddpm_tiers_path}. Run Cell 8.5.")

with open(ddpm_tiers_path, "r", encoding="utf-8") as f:
    ddpm_tiers_art = list(json.load(f))

if list(DDPM_TIERS_USED) != ddpm_tiers_art:
    raise RuntimeError(
        "[Cell9] DDPM_TIERS_USED mismatch vs locked artifact: "
        f"now={list(DDPM_TIERS_USED)} artifact={ddpm_tiers_art}"
    )

pre_policy_path = os.path.join(OUT_ART, "pre_cell9_mask_policy.json")
if not os.path.exists(pre_policy_path):
    raise RuntimeError(
        f"[Cell9] Missing PRE-CELL-9 policy artifact: {pre_policy_path}. "
        "Run the PRE-CELL-9 MASK POLICY CONTRACT first."
    )

with open(pre_policy_path, "r", encoding="utf-8") as f:
    pre_policy_art = json.load(f)

if not isinstance(pre_policy_art, dict):
    raise RuntimeError(f"[Cell9] pre_cell9_mask_policy.json must be a dict: {pre_policy_path}")

if str(pre_policy_art.get("version", "")).startswith("pre_cell9_mask_policy_v5_THESIS") is False:
    raise RuntimeError(
        "[Cell9] Expected pre_cell9_mask_policy_v5_THESIS policy contract. "
        f"Got version={pre_policy_art.get('version')!r}."
    )

_policy_leak = pre_policy_art.get("leakage_status", {})
if not isinstance(_policy_leak, dict):
    raise RuntimeError("[Cell9] pre_cell9_mask_policy.json missing leakage_status dict.")

for _k in [
    "uses_test_for_target_calibration",
    "uses_test_for_mask_model_fit",
    "uses_test_for_threshold_selection",
    "uses_test_for_candidate_selection",
    "uses_test_for_repair",
    "mutates_synthetic_artifact",
    "overwrites_final_artifact",
]:
    if bool(_policy_leak.get(_k, False)):
        raise RuntimeError(f"[Cell9] Pre-Cell-9 mask policy violates leakage rule: {_k}=True.")

if pre_policy_art.get("mask_target_base") != "train":
    raise RuntimeError(f"[Cell9] Canonical THESIS mask policy requires target_base=train. Got {pre_policy_art.get('mask_target_base')!r}.")

if pre_policy_art.get("mask_model_fit_split") != "train":
    raise RuntimeError(f"[Cell9] Canonical THESIS mask policy requires fit_split=train. Got {pre_policy_art.get('mask_model_fit_split')!r}.")

required_cfg_keys = [
    "mask_target_policy",
    "mask_target_base",
    "mask_target_policy_by_tier",
    "mask_target_q",
    "mask_target_q_by_tier",
    "mask_target_cap_at_base_mean",
    "mask_target_margin_default",
    "mask_target_margin_by_tier",
    "mask_target_floor_by_tier",
    "mask_target_ceiling_by_tier",
    "mask_force_mean_match",
    "mask_force_mean_mode",
    "fixA_max_flip_frac",
    "mask_model_fit_split",
    "mask_fit_tail_base",
    "mask_model_tail_frac",
]
missing_cfg = [k for k in required_cfg_keys if k not in CFG]
if missing_cfg:
    raise RuntimeError(f"[Cell9] Missing required mask-policy CFG keys: {missing_cfg}. Run PRE-CELL-9 MASK POLICY CONTRACT.")


# ------------------------------------------------------------
# 2) Tier sets and mask helpers
# ------------------------------------------------------------
allow_zwave = bool(CFG.get("allow_zwave", CFG.get("ddpm_allow_zwave", False)))

TIERS_CANONICAL = ["router", "ota", "zigbee", "zwave"]
TIERS_FULL = ["router", "ota", "zigbee"] + (["zwave"] if allow_zwave else [])
TIERS_DDPM = list(DDPM_TIERS_USED)

if len(TIERS_DDPM) != len(set(TIERS_DDPM)):
    raise RuntimeError(f"[Cell9] Duplicate entries in DDPM_TIERS_USED: {TIERS_DDPM}")

for t in TIERS_DDPM:
    if t not in TIERS_FULL:
        raise RuntimeError(f"[Cell9] DDPM tier '{t}' not allowed by TIERS_FULL={TIERS_FULL}")

if TIERS_DDPM != [t for t in TIERS_CANONICAL if t in TIERS_DDPM]:
    raise RuntimeError(f"[Cell9] DDPM tier order mismatch: {TIERS_DDPM}")

if TIERS_DDPM != ddpm_tiers_art:
    raise RuntimeError(f"[Cell9] DDPM tier artifact mismatch: {TIERS_DDPM} vs {ddpm_tiers_art}")


def _obs_col(tier: str) -> str:
    return f"{tier}__obs_present"


def _require_obs_cols(df_: pd.DataFrame, tiers: list, split_name: str) -> None:
    missing_cols = [_obs_col(t) for t in tiers if _obs_col(t) not in df_.columns]
    if missing_cols:
        raise RuntimeError(f"[Cell9] {split_name}: missing required obs columns: {missing_cols}")


for nm, part in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    _require_obs_cols(part, TIERS_FULL, nm)


def _mask_matrix(df_: pd.DataFrame, tiers: list, *, split_name: str) -> np.ndarray:
    _require_obs_cols(df_, tiers, split_name)
    mats = []
    for t in tiers:
        c = _obs_col(t)
        v = pd.to_numeric(df_[c], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        v = np.where(np.isfinite(v), v, 0.0)
        mats.append((v > 0.5).astype(np.int8, copy=False))
    out = np.stack(mats, axis=1).astype(np.int8, copy=False)
    if out.shape != (len(df_), len(tiers)):
        raise RuntimeError(f"[Cell9] mask matrix shape mismatch for {split_name}: got={out.shape}")
    return out


def _mask_rates(M: np.ndarray, tiers: list) -> dict:
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[1] != len(tiers):
        raise RuntimeError(f"[Cell9] _mask_rates shape mismatch: M={M.shape}, tiers={tiers}")
    return {t: float(np.mean(M[:, i] > 0)) for i, t in enumerate(tiers)}


def _sha256_jsonable(obj) -> str:
    s = json.dumps(_json_safe(obj), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2)


def _mean_obs(df_: pd.DataFrame, tier: str) -> float:
    c = _obs_col(tier)
    v = pd.to_numeric(df_[c], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    v = np.where(np.isfinite(v), v, 0.0)
    return float(np.mean(v > 0.5))


# ------------------------------------------------------------
# 3) Export real VAL/TEST masks as EVAL-only/reference globals
# ------------------------------------------------------------
REAL_VAL_MASKS_FULL = _mask_matrix(df_va, TIERS_FULL, split_name="VAL/REFERENCE_FULL")
REAL_VAL_MASKS_DDPM = _mask_matrix(df_va, TIERS_DDPM, split_name="VAL/REFERENCE_DDPM")
REAL_TEST_MASKS_FULL = _mask_matrix(df_te, TIERS_FULL, split_name="TEST/EVAL_ONLY_FULL")
REAL_TEST_MASKS_DDPM = _mask_matrix(df_te, TIERS_DDPM, split_name="TEST/EVAL_ONLY_DDPM")

globals()["REAL_VAL_MASKS_FULL"] = REAL_VAL_MASKS_FULL
globals()["REAL_VAL_MASKS_DDPM"] = REAL_VAL_MASKS_DDPM
globals()["REAL_TEST_MASKS_FULL"] = REAL_TEST_MASKS_FULL
globals()["REAL_TEST_MASKS_DDPM"] = REAL_TEST_MASKS_DDPM

log(f"[Cell9] Exported REAL_VAL masks: FULL={REAL_VAL_MASKS_FULL.shape} | DDPM={REAL_VAL_MASKS_DDPM.shape}")
log(f"[Cell9] Exported REAL_TEST masks as EVAL ONLY: FULL={REAL_TEST_MASKS_FULL.shape} | DDPM={REAL_TEST_MASKS_DDPM.shape}")
log(f"[Cell9] REAL_VAL mask rates FULL: {_mask_rates(REAL_VAL_MASKS_FULL, TIERS_FULL)}")
log(f"[Cell9] REAL_TEST mask rates FULL (EVAL ONLY): {_mask_rates(REAL_TEST_MASKS_FULL, TIERS_FULL)}")

log(
    "[Cell9][DBG:TRAIN] "
    + " | ".join([f"{_obs_col(t)}={_mean_obs(df_tr, t):.6f}" for t in TIERS_FULL])
)
log(
    "[Cell9][DBG:VAL]   "
    + " | ".join([f"{_obs_col(t)}={_mean_obs(df_va, t):.6f}" for t in TIERS_FULL])
)
log(
    "[Cell9][DBG:TEST]  "
    + " | ".join([f"{_obs_col(t)}={_mean_obs(df_te, t):.6f}" for t in TIERS_FULL])
)


# ------------------------------------------------------------
# 4) Choose leakage-safe fit split
# ------------------------------------------------------------
fit_split = str(CFG.get("mask_model_fit_split", "tail")).strip().lower()
tail_frac = float(CFG.get("mask_model_tail_frac", CFG.get("mask_target_tail_frac", 0.25)))
tail_frac = float(np.clip(tail_frac, 0.05, 0.95))
tail_base = str(CFG.get("mask_fit_tail_base", "val")).strip().lower()

if fit_split != "train":
    raise RuntimeError(
        "[Cell9] Canonical STUDY-THESIS mask policy requires mask_model_fit_split='train'. "
        f"Got {fit_split!r}. Do not use VAL/TEST/tail fitting in the clean run."
    )

if tail_base != "train":
    raise RuntimeError(
        "[Cell9] Canonical STUDY-THESIS mask policy requires mask_fit_tail_base='train'. "
        f"Got {tail_base!r}."
    )

if fit_split == "train":
    df_fit = df_tr.reset_index(drop=True)
    reg_fit = np.asarray(reg_tr, dtype=np.int64).reshape(-1)
    fit_rows_source = "train_all"

elif fit_split == "val":
    df_fit = df_va.reset_index(drop=True)
    reg_fit = np.asarray(reg_va, dtype=np.int64).reshape(-1)
    fit_rows_source = "val_all"

elif fit_split == "train_val":
    df_fit = pd.concat([df_tr, df_va], axis=0, ignore_index=True)
    reg_fit = np.concatenate([
        np.asarray(reg_tr, dtype=np.int64).reshape(-1),
        np.asarray(reg_va, dtype=np.int64).reshape(-1),
    ])
    fit_rows_source = "train_plus_val"

elif fit_split == "tail":
    if tail_base == "val":
        base_df = df_va
        base_reg = np.asarray(reg_va, dtype=np.int64).reshape(-1)
    elif tail_base == "train":
        base_df = df_tr
        base_reg = np.asarray(reg_tr, dtype=np.int64).reshape(-1)
    else:
        raise RuntimeError(f"[Cell9] Unsupported mask_fit_tail_base={tail_base!r}; use 'train' or 'val'.")

    if len(base_df) != len(base_reg):
        raise RuntimeError(
            f"[Cell9] tail base length mismatch: len(base_df)={len(base_df)} len(base_reg)={len(base_reg)}"
        )

    N0 = int(len(base_df))
    start = int(max(0, N0 - round(tail_frac * N0)))
    df_fit = base_df.iloc[start:].reset_index(drop=True)
    reg_fit = base_reg[start:].copy()
    fit_rows_source = f"{tail_base}_tail_start_{start}"

else:
    raise RuntimeError(
        f"[Cell9] Unknown mask_model_fit_split={fit_split!r}. "
        "Allowed: 'train', 'val', 'train_val', 'tail'."
    )

if len(df_fit) <= 0:
    raise RuntimeError("[Cell9] Fit split has zero rows.")

if len(df_fit) != len(reg_fit):
    raise RuntimeError(
        f"[Cell9] Fit split/reg length mismatch: len(df_fit)={len(df_fit)} len(reg_fit)={len(reg_fit)}"
    )

reg_va_arr = np.asarray(reg_va, dtype=np.int64).reshape(-1)
reg_te_arr = np.asarray(reg_te, dtype=np.int64).reshape(-1)

if len(reg_va_arr) != len(df_va):
    raise RuntimeError(f"[Cell9] reg_va length mismatch: len(reg_va)={len(reg_va_arr)} len(df_va)={len(df_va)}")
if len(reg_te_arr) != len(df_te):
    raise RuntimeError(f"[Cell9] reg_te length mismatch: len(reg_te)={len(reg_te_arr)} len(df_te)={len(df_te)}")

if reg_va_arr.size == 0:
    raise RuntimeError("[Cell9] reg_va is empty.")
if reg_te_arr.size == 0:
    raise RuntimeError("[Cell9] reg_te is empty.")

reg_va_unique = np.unique(reg_va_arr)
reg_te_unique = np.unique(reg_te_arr)
const_reg_va = bool(reg_va_unique.size == 1)
const_reg_te = bool(reg_te_unique.size == 1)

log(f"[Cell9] fit_split={fit_split} | fit_rows_source={fit_rows_source} | fit_rows={len(df_fit):,}")
if fit_split == "tail":
    log(f"[Cell9] tail_frac={tail_frac:.6f} | tail_base={tail_base}")

log(
    f"[Cell9] regimes | fit[min,max]={int(np.min(reg_fit))},{int(np.max(reg_fit))} | "
    f"val[min,max]={int(np.min(reg_va_arr))},{int(np.max(reg_va_arr))} | "
    f"test[min,max]={int(np.min(reg_te_arr))},{int(np.max(reg_te_arr))} | "
    f"const_reg_va={const_reg_va} | const_reg_te={const_reg_te}"
)


# ------------------------------------------------------------
# 5) Target policy config
# ------------------------------------------------------------
mask_target_policy = str(CFG.get("mask_target_policy", "by_tier")).strip().lower()
if mask_target_policy != "by_tier":
    raise RuntimeError(f"[Cell9] Expected CFG['mask_target_policy']='by_tier', got {mask_target_policy!r}.")

mask_target_base = str(CFG.get("mask_target_base", "train")).strip().lower()
if mask_target_base != "train":
    raise RuntimeError(
        "[Cell9] Canonical STUDY-THESIS mask policy requires mask_target_base='train'. "
        f"Got {mask_target_base!r}."
    )

if mask_target_base not in {"train", "val", "train_val"}:
    raise RuntimeError(
        f"[Cell9] mask_target_base must be 'train', 'val', or 'train_val'. "
        f"Got {mask_target_base!r}. TEST is forbidden."
    )

if mask_target_base == "train":
    df_target_base = df_tr
elif mask_target_base == "val":
    df_target_base = df_va
elif mask_target_base == "train_val":
    df_target_base = pd.concat([df_tr, df_va], axis=0, ignore_index=True)
else:
    raise RuntimeError(f"[Cell9] Unreachable mask_target_base={mask_target_base!r}")

mask_target_window = int(CFG.get("mask_target_window", 1800))
if mask_target_window <= 0:
    raise RuntimeError(f"[Cell9] mask_target_window must be >0, got {mask_target_window}")

mask_target_tail_frac = float(CFG.get("mask_target_tail_frac", 0.25))
mask_target_tail_frac = float(np.clip(mask_target_tail_frac, 0.05, 0.95))

policy_by_tier = dict(CFG.get("mask_target_policy_by_tier", {}) or {})
q_default = float(CFG.get("mask_target_q", 0.25))
q_by_tier = dict(CFG.get("mask_target_q_by_tier", {}) or {})
cap_at_base_mean = bool(CFG.get("mask_target_cap_at_base_mean", True))
margin_default = float(CFG.get("mask_target_margin_default", 0.0))
margin_by_tier = dict(CFG.get("mask_target_margin_by_tier", {}) or {})
floor_by_tier = dict(CFG.get("mask_target_floor_by_tier", {}) or {})
ceiling_by_tier = dict(CFG.get("mask_target_ceiling_by_tier", {}) or {})
ota_force_ones_thr = float(CFG.get("ota_force_ones_thr", 0.999))

allowed_policies = {"quantile_tail", "force_ones_if_high", "conservative_cap"}
missing_tier_policies = [t for t in TIERS_FULL if t not in policy_by_tier]
if missing_tier_policies:
    raise RuntimeError(f"[Cell9] Missing mask target policies for tiers: {missing_tier_policies}")

for t in TIERS_FULL:
    pol = str(policy_by_tier[t]).strip().lower()
    if pol not in allowed_policies:
        raise RuntimeError(f"[Cell9] Unsupported mask target policy for tier={t!r}: {pol!r}")
    policy_by_tier[t] = pol


def _base_mean(df_base: pd.DataFrame, tier: str) -> float:
    c = _obs_col(tier)
    if c not in df_base.columns:
        raise RuntimeError(f"[Cell9] Missing target-base obs column: {c}")
    v = pd.to_numeric(df_base[c], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    v = np.where(np.isfinite(v), v, 0.0)
    return float(np.mean(v > 0.5))


def _rolling_coverage_quantile(
    df_base: pd.DataFrame,
    tier: str,
    q: float,
    window: int,
    tail_frac_local: float,
):
    c = _obs_col(tier)
    if c not in df_base.columns:
        raise RuntimeError(f"[Cell9] Missing rolling target obs column: {c}")

    s = pd.to_numeric(df_base[c], errors="coerce").fillna(0.0).astype(float)
    rm = s.rolling(window, min_periods=window).mean().dropna()

    if len(rm) < 10:
        base_mean_local = float(np.mean(s.values > 0.5))
        return base_mean_local, {
            "n_roll": int(len(rm)),
            "n_tail": 0,
            "used_tail": False,
            "fallback": "base_mean_due_to_short_rolling_series",
        }

    n_tail = max(10, int(round(len(rm) * float(tail_frac_local))))
    n_tail = min(n_tail, len(rm))
    rm_use = rm.iloc[-n_tail:]

    val = float(np.quantile(rm_use.values, float(q)))
    return val, {
        "n_roll": int(len(rm)),
        "n_tail": int(len(rm_use)),
        "used_tail": True,
        "fallback": None,
    }


def _apply_floor_ceiling(x: float, tier: str) -> float:
    x = float(x)

    floor_v = floor_by_tier.get(tier, None)
    ceil_v = ceiling_by_tier.get(tier, None)

    if floor_v is not None:
        x = max(x, float(floor_v))
    if ceil_v is not None:
        x = min(x, float(ceil_v))

    return float(np.clip(x, 1e-6, 1.0 - 1e-6))


def _compute_target_for_tier(df_base: pd.DataFrame, tier: str):
    pol = str(policy_by_tier.get(tier, "quantile_tail")).strip().lower()
    base_mean_local = _base_mean(df_base, tier)
    q_t = float(np.clip(float(q_by_tier.get(tier, q_default)), 0.01, 0.99))
    margin_t = float(margin_by_tier.get(tier, margin_default))

    if pol == "force_ones_if_high":
        if base_mean_local >= ota_force_ones_thr:
            val = 0.999999
            action = "forced_to_near_one"
        else:
            val = base_mean_local
            action = "used_base_mean_below_force_threshold"

        meta = {
            "policy": pol,
            "base_mean": float(base_mean_local),
            "ota_force_ones_thr": float(ota_force_ones_thr),
            "action": action,
            "q_used": None,
            "rolling_tail_quantile": None,
            "capped_at_base_mean": False,
            "margin": 0.0,
            "pre_floor_ceiling_target": float(val),
        }
        return _apply_floor_ceiling(val, tier), meta

    roll_q, roll_meta = _rolling_coverage_quantile(
        df_base=df_base,
        tier=tier,
        q=q_t,
        window=mask_target_window,
        tail_frac_local=mask_target_tail_frac,
    )

    capped = False
    if pol == "quantile_tail":
        val = float(roll_q)

    elif pol == "conservative_cap":
        val = float(roll_q)
        if cap_at_base_mean and val > base_mean_local:
            val = float(base_mean_local)
            capped = True
        val = float(val - margin_t)

    else:
        raise RuntimeError(f"[Cell9] Unsupported policy={pol!r} for tier={tier!r}")

    meta = {
        "policy": pol,
        "base_mean": float(base_mean_local),
        "q_used": float(q_t),
        "rolling_tail_quantile": float(roll_q),
        "capped_at_base_mean": bool(capped),
        "margin": float(margin_t) if pol == "conservative_cap" else 0.0,
        "pre_floor_ceiling_target": float(val),
        **roll_meta,
    }

    return _apply_floor_ceiling(val, tier), meta


target_pi_by_tier = {}
target_meta_by_tier = {}
for t in TIERS_FULL:
    target_pi_by_tier[t], target_meta_by_tier[t] = _compute_target_for_tier(df_target_base, t)

globals()["target_pi_by_tier"] = dict(target_pi_by_tier)
globals()["target_meta_by_tier"] = dict(target_meta_by_tier)

log(
    "[Cell9] Target policy locked: "
    f"base={mask_target_base} | window={mask_target_window} | tail_frac={mask_target_tail_frac:.3f} | "
    f"targets={{{', '.join([f'{k}: {target_pi_by_tier[k]:.6f}' for k in TIERS_FULL])}}}"
)


# ------------------------------------------------------------
# 6) Joint episode helpers
# ------------------------------------------------------------
def _state_code_row(row: np.ndarray) -> int:
    code = 0
    for i, bit in enumerate(np.asarray(row, dtype=np.int8).reshape(-1).tolist()):
        code |= (int(bit > 0) << i)
    return int(code)


def _extract_joint_episodes(M: np.ndarray, reg_arr: np.ndarray):
    """
    Extract joint mask episodes.

    Split when:
    - joint mask state changes, or
    - regime changes.
    """
    M = np.asarray(M, dtype=np.int8)
    reg_arr = np.asarray(reg_arr, dtype=np.int64).reshape(-1)

    if M.ndim != 2:
        raise RuntimeError(f"[Cell9] Episode extraction expects 2D mask matrix, got {M.shape}")
    if M.shape[0] != reg_arr.shape[0]:
        raise RuntimeError("[Cell9] Joint episode extraction length mismatch.")

    episodes = []
    if M.shape[0] == 0:
        return episodes

    state_codes = np.array([_state_code_row(M[i]) for i in range(M.shape[0])], dtype=np.int64)

    s = 0
    for i in range(1, M.shape[0] + 1):
        end_run = (
            i == M.shape[0]
            or state_codes[i] != state_codes[s]
            or int(reg_arr[i]) != int(reg_arr[s])
        )
        if end_run:
            episodes.append({
                "start": int(s),
                "end": int(i),
                "len": int(i - s),
                "state": M[s].copy(),
                "state_code": int(state_codes[s]),
                "reg": int(reg_arr[s]),
            })
            s = i

    return episodes


def _episode_runs(mask_1d: np.ndarray):
    x = np.asarray(mask_1d, dtype=np.int8).reshape(-1)
    out = []
    if x.size == 0:
        return out

    s = 0
    cur = int(x[0])
    for i in range(1, x.size + 1):
        if i == x.size or int(x[i]) != cur:
            out.append((int(s), int(i), int(cur)))
            if i < x.size:
                s = i
                cur = int(x[i])
    return out


def _spread_pick(cand: np.ndarray, flips: int) -> np.ndarray:
    cand = np.asarray(cand, dtype=np.int64).reshape(-1)
    flips = int(flips)

    if cand.size == 0 or flips <= 0:
        return np.zeros(0, dtype=np.int64)

    if flips >= cand.size:
        return cand.copy()

    pos = np.linspace(0, cand.size - 1, num=flips)
    pos = np.unique(np.round(pos).astype(int))

    if pos.size < flips:
        used = set(pos.tolist())
        extra = [i for i in range(cand.size) if i not in used][: (flips - pos.size)]
        if extra:
            pos = np.concatenate([pos, np.asarray(extra, dtype=int)])

    pos = np.sort(pos[:flips])
    return cand[pos]


def _fixA_mean_correct(mask_1d: np.ndarray, target_pi: float, *, mode: str, max_flip_frac: float):
    """
    Tiny boundary-biased cleanup only. This is not the primary calibrator.
    """
    m = np.asarray(mask_1d, dtype=np.int8).copy()
    N = int(m.size)

    if N <= 0:
        return m, 0, 0.0, 0.0

    target_pi = float(np.clip(target_pi, 0.0, 1.0))
    cur = float(np.mean(m))
    need = int(round((target_pi - cur) * N))

    max_flips = int(np.floor(float(max_flip_frac) * N))
    if max_flips <= 0 or need == 0:
        return m, 0, cur, cur

    flips = int(min(abs(need), max_flips))
    if flips <= 0:
        return m, 0, cur, cur

    mode = str(mode).lower()

    if mode == "boundary":
        left = np.empty(N, dtype=np.int8)
        left[0] = m[0]
        left[1:] = m[:-1]

        right = np.empty(N, dtype=np.int8)
        right[-1] = m[-1]
        right[:-1] = m[1:]

        boundary = (left != m) | (right != m)
    else:
        boundary = None

    if need > 0:
        idx = np.flatnonzero(m == 0)
        if idx.size == 0:
            return m, 0, cur, cur
        cand = idx[boundary[idx]] if boundary is not None else idx
        if cand.size == 0:
            cand = idx
        sel = _spread_pick(cand, min(flips, cand.size))
        m[sel] = 1
        actual = int(sel.size)

    else:
        idx = np.flatnonzero(m == 1)
        if idx.size == 0:
            return m, 0, cur, cur
        cand = idx[boundary[idx]] if boundary is not None else idx
        if cand.size == 0:
            cand = idx
        sel = _spread_pick(cand, min(flips, cand.size))
        m[sel] = 0
        actual = int(sel.size)

    after = float(np.mean(m))
    return m, actual, cur, after


def _reduce_positive_coverage_by_runs(
    mask_1d: np.ndarray,
    target_pi: float,
    *,
    prefer_short_first: bool = True,
):
    """
    Target-aware positive-coverage reduction.

    Strategy:
    - remove whole short positive runs when helpful;
    - finish with boundary trimming within one run if needed.
    """
    m = np.asarray(mask_1d, dtype=np.int8).copy()
    N = int(m.size)

    if N <= 0:
        return m, {
            "applied": False,
            "full_runs_zeroed": 0,
            "boundary_trim_cells": 0,
            "pi_before": 0.0,
            "pi_after": 0.0,
        }

    tgt = float(np.clip(target_pi, 0.0, 1.0))
    before = float(np.mean(m))

    if before <= tgt + 1e-12:
        return m, {
            "applied": False,
            "full_runs_zeroed": 0,
            "boundary_trim_cells": 0,
            "pi_before": before,
            "pi_after": before,
        }

    desired_ones = int(round(tgt * N))
    current_ones = int(np.sum(m))
    excess = int(max(0, current_ones - desired_ones))

    if excess <= 0:
        return m, {
            "applied": False,
            "full_runs_zeroed": 0,
            "boundary_trim_cells": 0,
            "pi_before": before,
            "pi_after": before,
        }

    pos_runs = [(s, e, e - s) for (s, e, v) in _episode_runs(m) if int(v) == 1]
    if not pos_runs:
        return m, {
            "applied": False,
            "full_runs_zeroed": 0,
            "boundary_trim_cells": 0,
            "pi_before": before,
            "pi_after": before,
        }

    if prefer_short_first:
        order = np.argsort([L for (_, _, L) in pos_runs])
    else:
        order = np.arange(len(pos_runs))

    removed = 0
    full_runs_zeroed = 0

    for idx in order:
        s, e, L = pos_runs[int(idx)]

        if removed >= excess:
            break

        before_gap = abs((current_ones - removed) - desired_ones)
        after_gap = abs((current_ones - removed - L) - desired_ones)

        if L <= max(1, excess - removed) or after_gap <= before_gap:
            m[s:e] = 0
            removed += int(L)
            full_runs_zeroed += 1

    current_after_full = int(np.sum(m))
    excess_after_full = int(max(0, current_after_full - desired_ones))

    boundary_trim_cells = 0
    if excess_after_full > 0:
        remaining_pos_runs = [(s, e, e - s) for (s, e, v) in _episode_runs(m) if int(v) == 1]
        if remaining_pos_runs:
            remaining_pos_runs = sorted(remaining_pos_runs, key=lambda z: z[2])
            s, e, L = remaining_pos_runs[0]
            trim = int(min(excess_after_full, L))
            m[e - trim:e] = 0
            boundary_trim_cells = trim

    after = float(np.mean(m))
    return m, {
        "applied": True,
        "full_runs_zeroed": int(full_runs_zeroed),
        "boundary_trim_cells": int(boundary_trim_cells),
        "pi_before": float(before),
        "pi_after": float(after),
    }


# ------------------------------------------------------------
# 7) Fit joint DDPM-tier episode pool
# ------------------------------------------------------------
M_fit_ddpm = _mask_matrix(df_fit, TIERS_DDPM, split_name=f"FIT/{fit_rows_source}/DDPM")
N_fit = int(M_fit_ddpm.shape[0])

if N_fit <= 0:
    raise RuntimeError("[Cell9] Fit split has zero rows for joint mask bootstrap.")

episodes = _extract_joint_episodes(M_fit_ddpm, reg_fit)
if not episodes:
    raise RuntimeError("[Cell9] No joint mask episodes extracted.")

ep_pool_by_reg = {}
for ep in episodes:
    ep_pool_by_reg.setdefault(int(ep["reg"]), []).append(ep)

global_pool = list(episodes)

episode_len_values = np.array([ep["len"] for ep in episodes], dtype=np.int64)
episode_state_counts = {}
for ep in episodes:
    episode_state_counts[str(int(ep["state_code"]))] = episode_state_counts.get(str(int(ep["state_code"])), 0) + 1

log(
    "[Cell9] Joint episode pool: "
    f"episodes={len(episodes):,} | regs={sorted(ep_pool_by_reg.keys())} | "
    f"len_median={float(np.median(episode_len_values)):.2f} | "
    f"len_p95={float(np.quantile(episode_len_values, 0.95)):.2f}"
)


def _sample_episode_for_reg(r: int, rng_local: np.random.Generator):
    r = int(r)
    pool = ep_pool_by_reg.get(r)
    if not pool:
        pool = global_pool
    idx = int(rng_local.integers(0, len(pool)))
    return pool[idx]


# ------------------------------------------------------------
# 8) Generate and calibrate masks for any horizon
# ------------------------------------------------------------
def _generate_syn_masks_for_horizon(
    *,
    horizon_name: str,
    N_h: int,
    reg_h: np.ndarray,
    rng_local: np.random.Generator,
):
    """
    Generate synthetic DDPM-tier and FULL-tier masks for a target horizon.

    Uses:
    - TRAIN/VAL-fitted episode pool.
    - Horizon regime sequence only.
    - No horizon real mask values.
    """
    N_h = int(N_h)
    reg_h = np.asarray(reg_h, dtype=np.int64).reshape(-1)

    if N_h <= 0:
        raise RuntimeError(f"[Cell9] {horizon_name}: horizon has zero rows.")
    if reg_h.shape[0] != N_h:
        raise RuntimeError(
            f"[Cell9] {horizon_name}: regime length mismatch: reg={reg_h.shape[0]} N={N_h}"
        )

    M_ddpm = np.zeros((N_h, len(TIERS_DDPM)), dtype=np.int8)

    pos = 0
    sampled_episode_count = 0

    while pos < N_h:
        reg_here = int(reg_h[pos])
        ep = _sample_episode_for_reg(reg_here, rng_local)

        L = int(max(1, ep["len"]))

        next_change = pos + 1
        while next_change < N_h and int(reg_h[next_change]) == reg_here:
            next_change += 1

        take = min(L, N_h - pos, next_change - pos)

        if take <= 0:
            raise RuntimeError(
                f"[Cell9] {horizon_name}: invalid episode take length at pos={pos}: "
                f"L={L}, next_change={next_change}, N_h={N_h}"
            )

        M_ddpm[pos:pos + take, :] = ep["state"][None, :]
        pos += take
        sampled_episode_count += 1

    bootstrap_rates = _mask_rates(M_ddpm, TIERS_DDPM)

    mask_force_mean_match = bool(CFG.get("mask_force_mean_match", True))
    mask_force_mean_mode = str(CFG.get("mask_force_mean_mode", "boundary")).strip().lower()
    fixA_max_flip_frac = float(np.clip(float(CFG.get("fixA_max_flip_frac", 0.002)), 0.0, 0.05))
    target_material_gap = float(CFG.get("mask_target_material_gap", 0.005))
    target_tolerance_warn = float(CFG.get("mask_target_tolerance_warn", 0.015))

    per_tier_info = {}

    for j, t in enumerate(TIERS_DDPM):
        seq = M_ddpm[:, j].copy()
        pi_tgt = float(target_pi_by_tier[t])
        pol_t = str(policy_by_tier.get(t, "quantile_tail")).lower()

        pi_initial = float(np.mean(seq))

        run_cal = {
            "applied": False,
            "full_runs_zeroed": 0,
            "boundary_trim_cells": 0,
            "pi_before": float(pi_initial),
            "pi_after": float(pi_initial),
        }

        if pi_initial > pi_tgt + target_material_gap:
            seq2, run_cal = _reduce_positive_coverage_by_runs(
                seq,
                target_pi=pi_tgt,
                prefer_short_first=True,
            )
            seq = seq2
            if run_cal["applied"]:
                log(
                    f"[Cell9][{horizon_name}][RUN-CAL] {t}: target={pi_tgt:.6f} | "
                    f"before={run_cal['pi_before']:.6f} | after={run_cal['pi_after']:.6f} | "
                    f"full_runs_zeroed={run_cal['full_runs_zeroed']} | "
                    f"boundary_trim_cells={run_cal['boundary_trim_cells']}"
                )

        fixA_flips = 0
        before_fixA = float(np.mean(seq))

        if mask_force_mean_match:
            seq2, fixA_flips, bf, af = _fixA_mean_correct(
                seq,
                target_pi=pi_tgt,
                mode=mask_force_mean_mode,
                max_flip_frac=fixA_max_flip_frac,
            )
            seq = seq2
            if fixA_flips > 0:
                log(
                    f"[Cell9][{horizon_name}][FixA] {t}: target={pi_tgt:.6f} | "
                    f"before={bf:.6f} | after={af:.6f} | flips={fixA_flips}"
                )

        pi_after = float(np.mean(seq))
        abs_target_err = float(abs(pi_after - pi_tgt))

        if abs_target_err > target_tolerance_warn:
            log(
                f"[Cell9][WARN][{horizon_name}] {t}: synthetic mask rate remains far from leakage-safe target | "
                f"rate={pi_after:.6f} target={pi_tgt:.6f} abs_err={abs_target_err:.6f}"
            )

        M_ddpm[:, j] = seq.astype(np.int8, copy=False)

        per_tier_info[t] = {
            "policy": pol_t,
            "pi_target": float(pi_tgt),
            "pi_initial_bootstrap": float(pi_initial),
            "run_calibration": dict(run_cal),
            "pi_before_fixA": float(before_fixA),
            "fixA_flips": int(fixA_flips),
            "pi_after": float(pi_after),
            "abs_target_error": float(abs_target_err),
            "target_meta": dict(target_meta_by_tier.get(t, {})),
        }

    M_full = np.zeros((N_h, len(TIERS_FULL)), dtype=np.int8)
    for j, t in enumerate(TIERS_DDPM):
        M_full[:, TIERS_FULL.index(t)] = M_ddpm[:, j]

    rates_full = _mask_rates(M_full, TIERS_FULL)
    rates_ddpm = _mask_rates(M_ddpm, TIERS_DDPM)

    log(f"[Cell9] Generated {horizon_name} SYN masks: FULL={M_full.shape} | DDPM={M_ddpm.shape}")
    log(f"[Cell9] {horizon_name} SYN mask rates FULL: {rates_full}")
    log(f"[Cell9] {horizon_name} SYN mask rates DDPM: {rates_ddpm}")

    meta = {
        "horizon": str(horizon_name),
        "N": int(N_h),
        "sampled_episode_count": int(sampled_episode_count),
        "bootstrap_rates_ddpm": dict(bootstrap_rates),
        "syn_rates_full": dict(rates_full),
        "syn_rates_ddpm": dict(rates_ddpm),
        "per_tier": dict(per_tier_info),
        "reg_unique": [int(x) for x in np.unique(reg_h).tolist()],
        "const_reg": bool(np.unique(reg_h).size == 1),
    }

    return M_full, M_ddpm, meta


SYN_MASKS_VAL_FULL, SYN_MASKS_VAL_DDPM, val_mask_generation_meta = _generate_syn_masks_for_horizon(
    horizon_name="VAL",
    N_h=len(df_va),
    reg_h=reg_va_arr,
    rng_local=rng_val,
)

SYN_MASKS_TEST_FULL, SYN_MASKS_TEST_DDPM, test_mask_generation_meta = _generate_syn_masks_for_horizon(
    horizon_name="TEST",
    N_h=len(df_te),
    reg_h=reg_te_arr,
    rng_local=rng_test,
)

# Backward-compatible aliases expected by existing downstream cells.
SYN_MASKS_FULL = SYN_MASKS_TEST_FULL
SYN_MASKS_DDPM = SYN_MASKS_TEST_DDPM

globals()["SYN_MASKS_VAL_FULL"] = SYN_MASKS_VAL_FULL
globals()["SYN_MASKS_VAL_DDPM"] = SYN_MASKS_VAL_DDPM
globals()["SYN_MASKS_TEST_FULL"] = SYN_MASKS_TEST_FULL
globals()["SYN_MASKS_TEST_DDPM"] = SYN_MASKS_TEST_DDPM
globals()["SYN_MASKS_FULL"] = SYN_MASKS_FULL
globals()["SYN_MASKS_DDPM"] = SYN_MASKS_DDPM

syn_val_rates_full = _mask_rates(SYN_MASKS_VAL_FULL, TIERS_FULL)
syn_val_rates_ddpm = _mask_rates(SYN_MASKS_VAL_DDPM, TIERS_DDPM)
syn_test_rates_full = _mask_rates(SYN_MASKS_TEST_FULL, TIERS_FULL)
syn_test_rates_ddpm = _mask_rates(SYN_MASKS_TEST_DDPM, TIERS_DDPM)


# ------------------------------------------------------------
# 9) Joint structure diagnostics
# ------------------------------------------------------------
def _pairwise_jaccard(M: np.ndarray, tiers: list) -> dict:
    M = np.asarray(M, dtype=np.int8)
    out = {}
    for i in range(len(tiers)):
        for j in range(i + 1, len(tiers)):
            a = M[:, i] > 0
            b = M[:, j] > 0
            den = int(np.sum(a | b))
            val = float(np.sum(a & b) / den) if den > 0 else 0.0
            out[f"{tiers[i]}__{tiers[j]}"] = val
    return out


def _state_distribution(M: np.ndarray) -> dict:
    M = np.asarray(M, dtype=np.int8)
    if M.size == 0:
        return {}
    codes = np.array([_state_code_row(M[i]) for i in range(M.shape[0])], dtype=np.int64)
    vals, cnts = np.unique(codes, return_counts=True)
    n = float(len(codes))
    return {str(int(v)): float(c / n) for v, c in zip(vals, cnts)}


fit_rates_ddpm = _mask_rates(M_fit_ddpm, TIERS_DDPM)
fit_jaccard_ddpm = _pairwise_jaccard(M_fit_ddpm, TIERS_DDPM)
fit_state_dist = _state_distribution(M_fit_ddpm)

syn_val_jaccard_ddpm = _pairwise_jaccard(SYN_MASKS_VAL_DDPM, TIERS_DDPM)
syn_test_jaccard_ddpm = _pairwise_jaccard(SYN_MASKS_TEST_DDPM, TIERS_DDPM)
syn_val_state_dist = _state_distribution(SYN_MASKS_VAL_DDPM)
syn_test_state_dist = _state_distribution(SYN_MASKS_TEST_DDPM)

real_val_rates_full = _mask_rates(REAL_VAL_MASKS_FULL, TIERS_FULL)
real_val_rates_ddpm = _mask_rates(REAL_VAL_MASKS_DDPM, TIERS_DDPM)
real_test_rates_full = _mask_rates(REAL_TEST_MASKS_FULL, TIERS_FULL)
real_test_rates_ddpm = _mask_rates(REAL_TEST_MASKS_DDPM, TIERS_DDPM)

real_val_jaccard_ddpm = _pairwise_jaccard(REAL_VAL_MASKS_DDPM, TIERS_DDPM)
real_test_jaccard_ddpm = _pairwise_jaccard(REAL_TEST_MASKS_DDPM, TIERS_DDPM)

coverage_drift_val_eval = {
    t: float(syn_val_rates_full[t] - real_val_rates_full[t])
    for t in TIERS_FULL
}
coverage_drift_test_eval = {
    t: float(syn_test_rates_full[t] - real_test_rates_full[t])
    for t in TIERS_FULL
}

jaccard_drift_val_eval = {
    k: float(syn_val_jaccard_ddpm[k] - real_val_jaccard_ddpm.get(k, 0.0))
    for k in syn_val_jaccard_ddpm
}
jaccard_drift_test_eval = {
    k: float(syn_test_jaccard_ddpm[k] - real_test_jaccard_ddpm.get(k, 0.0))
    for k in syn_test_jaccard_ddpm
}

log(f"[Cell9] Fit mask rates DDPM: {fit_rates_ddpm}")
log(f"[Cell9] Fit-vs-SYN_VAL joint Jaccard DDPM: fit={fit_jaccard_ddpm} | syn_val={syn_val_jaccard_ddpm}")
log(f"[Cell9] Fit-vs-SYN_TEST joint Jaccard DDPM: fit={fit_jaccard_ddpm} | syn_test={syn_test_jaccard_ddpm}")
log(f"[Cell9] VAL diagnostic coverage drift SYN_VAL-REAL_VAL: {coverage_drift_val_eval}")
log(f"[Cell9] TEST EVAL-only coverage drift SYN_TEST-REAL_TEST: {coverage_drift_test_eval}")


# ------------------------------------------------------------
# 10) Persist leakage-safe model artifact and separate eval diagnostics
# ------------------------------------------------------------
mask_force_mean_match = bool(CFG.get("mask_force_mean_match", True))
mask_force_mean_mode = str(CFG.get("mask_force_mean_mode", "boundary")).strip().lower()
fixA_max_flip_frac = float(np.clip(float(CFG.get("fixA_max_flip_frac", 0.002)), 0.0, 0.05))
target_material_gap = float(CFG.get("mask_target_material_gap", 0.005))
target_tolerance_warn = float(CFG.get("mask_target_tolerance_warn", 0.015))

tier_mask_model = {
    "version": "v11_leakage_safe_joint_episode_bootstrap_val_and_test_generation",
    "seed_fit": int(seed + 900),
    "seed_val_generation": int(seed + 901),
    "seed_test_generation": int(seed + 902),

    "tiers_full": list(TIERS_FULL),
    "tiers_ddpm": list(TIERS_DDPM),
    "allow_zwave": bool(allow_zwave),

    "fit_split": str(fit_split),
    "fit_rows_source": str(fit_rows_source),
    "fit_rows": int(len(df_fit)),
    "tail_frac": float(tail_frac) if fit_split == "tail" else None,
    "tail_base": str(tail_base) if fit_split == "tail" else None,

    "generation_mode": "joint_episode_bootstrap_regime_aware_then_target_run_calibration",
    "episode_split_policy": "split_on_joint_state_or_regime_change",
    "joint_episode_count": int(len(episodes)),
    "episode_length_summary": {
        "min": int(np.min(episode_len_values)),
        "median": float(np.median(episode_len_values)),
        "p95": float(np.quantile(episode_len_values, 0.95)),
        "max": int(np.max(episode_len_values)),
    },
    "episode_state_counts": dict(episode_state_counts),

    "mask_target_policy": "by_tier",
    "mask_target_base": str(mask_target_base),
    "mask_target_window": int(mask_target_window),
    "mask_target_tail_frac": float(mask_target_tail_frac),
    "mask_target_policy_by_tier": dict(policy_by_tier),
    "target_pi_by_tier": dict(target_pi_by_tier),
    "target_meta_by_tier": dict(target_meta_by_tier),

    "mask_force_mean_match": bool(mask_force_mean_match),
    "mask_force_mean_mode": str(mask_force_mean_mode),
    "fixA_max_flip_frac": float(fixA_max_flip_frac),
    "target_material_gap": float(target_material_gap),
    "target_tolerance_warn": float(target_tolerance_warn),

    "fit_rates_ddpm": dict(fit_rates_ddpm),

    "syn_val_generation_meta": dict(val_mask_generation_meta),
    "syn_test_generation_meta": dict(test_mask_generation_meta),

    "syn_val_rates_full": dict(syn_val_rates_full),
    "syn_val_rates_ddpm": dict(syn_val_rates_ddpm),
    "syn_test_rates_full": dict(syn_test_rates_full),
    "syn_test_rates_ddpm": dict(syn_test_rates_ddpm),

    "pairwise_jaccard_fit_ddpm": dict(fit_jaccard_ddpm),
    "pairwise_jaccard_syn_val_ddpm": dict(syn_val_jaccard_ddpm),
    "pairwise_jaccard_syn_test_ddpm": dict(syn_test_jaccard_ddpm),
    "state_distribution_fit_ddpm": dict(fit_state_dist),
    "state_distribution_syn_val_ddpm": dict(syn_val_state_dist),
    "state_distribution_syn_test_ddpm": dict(syn_test_state_dist),

    "pre_cell9_policy_artifact": os.path.basename(pre_policy_path),
    "pre_cell9_policy_sha256": _sha256_jsonable(pre_policy_art),

    "leakage_policy": {
        "test_real_values_used_for_fitting": False,
        "test_real_values_used_for_target_calibration": False,
        "test_real_values_used_for_generation": False,
        "test_horizon_length_and_regime_used_for_synthetic_generation": True,
        "test_real_values_used_for_model_selection": False,
        "test_real_diagnostics_saved_separately": True,
        "val_used_for_selection_downstream": True,
        "val_masks_generated_for_A1_VAL": True,
    },

    "notes": [
        f"Targets are calibrated from {mask_target_base.upper()} only; TEST is forbidden.",
        "Mask episodes are fitted from TRAIN/VAL-only split selected by CFG.",
        "Synthetic VAL masks are generated for A1_VAL and candidate-selection baselines.",
        "Synthetic TEST masks are generated for A1_TEST/A2 final evaluation.",
        "Real TEST statistics are excluded from this model artifact. Synthetic TEST-horizon generation metadata is included because it describes the synthetic output, not real TEST values.",
        "Real TEST diagnostics are saved only in tier_mask_eval_diagnostics.json.",
    ],
}

tier_mask_model_path = os.path.join(OUT_ART, "tier_mask_model.json")
_write_json(tier_mask_model_path, tier_mask_model)

tier_mask_eval = {
    "version": "v11_val_and_test_eval_diagnostics",
    "eval_only": True,
    "warning": (
        "This artifact contains VAL/TEST real-mask comparison diagnostics. "
        "Do not load it for generation, fitting, target calibration, or model selection."
    ),
    "tiers_full": list(TIERS_FULL),
    "tiers_ddpm": list(TIERS_DDPM),

    "val": {
        "real_rates_full": dict(real_val_rates_full),
        "real_rates_ddpm": dict(real_val_rates_ddpm),
        "syn_rates_full": dict(syn_val_rates_full),
        "syn_rates_ddpm": dict(syn_val_rates_ddpm),
        "coverage_drift_syn_minus_real_full": dict(coverage_drift_val_eval),
        "pairwise_jaccard_real_ddpm": dict(real_val_jaccard_ddpm),
        "pairwise_jaccard_syn_ddpm": dict(syn_val_jaccard_ddpm),
        "pairwise_jaccard_drift_syn_minus_real_ddpm": dict(jaccard_drift_val_eval),
        "const_reg": bool(const_reg_va),
        "reg_unique": [int(x) for x in reg_va_unique.tolist()],
    },

    "test_eval_only": {
        "real_rates_full": dict(real_test_rates_full),
        "real_rates_ddpm": dict(real_test_rates_ddpm),
        "syn_rates_full": dict(syn_test_rates_full),
        "syn_rates_ddpm": dict(syn_test_rates_ddpm),
        "coverage_drift_syn_minus_real_full": dict(coverage_drift_test_eval),
        "pairwise_jaccard_real_ddpm": dict(real_test_jaccard_ddpm),
        "pairwise_jaccard_syn_ddpm": dict(syn_test_jaccard_ddpm),
        "pairwise_jaccard_drift_syn_minus_real_ddpm": dict(jaccard_drift_test_eval),
        "const_reg": bool(const_reg_te),
        "reg_unique": [int(x) for x in reg_te_unique.tolist()],
    },

    "leakage_policy": {
        "test_real_values_used_for_fitting": False,
        "test_real_values_used_for_target_calibration": False,
        "test_real_values_used_for_generation": False,
        "test_horizon_length_and_regime_used_for_synthetic_generation": True,
        "test_real_values_used_for_model_selection": False,
        "test_diagnostics_eval_only": True,
    },
}

tier_mask_eval_path = os.path.join(OUT_ART, "tier_mask_eval_diagnostics.json")
_write_json(tier_mask_eval_path, tier_mask_eval)

log(f"[Cell9] Saved leakage-safe tier mask model: {tier_mask_model_path}")
log(f"[Cell9] Saved VAL/TEST eval diagnostics: {tier_mask_eval_path}")


# ------------------------------------------------------------
# 11) Persist mask arrays for reproducibility
# ------------------------------------------------------------
mask_npz_path = os.path.join(OUT_ART, "cell9_synthetic_mask_arrays_v11_THESIS.npz")
np.savez_compressed(
    mask_npz_path,
    SYN_MASKS_VAL_FULL=SYN_MASKS_VAL_FULL,
    SYN_MASKS_VAL_DDPM=SYN_MASKS_VAL_DDPM,
    SYN_MASKS_TEST_FULL=SYN_MASKS_TEST_FULL,
    SYN_MASKS_TEST_DDPM=SYN_MASKS_TEST_DDPM,
    REAL_VAL_MASKS_FULL=REAL_VAL_MASKS_FULL,
    REAL_VAL_MASKS_DDPM=REAL_VAL_MASKS_DDPM,
    REAL_TEST_MASKS_FULL=REAL_TEST_MASKS_FULL,
    REAL_TEST_MASKS_DDPM=REAL_TEST_MASKS_DDPM,
)

log(f"[Cell9] Saved compressed mask arrays: {mask_npz_path}")


# ------------------------------------------------------------
# 12) Driver matrices — strict, no silent reconstruction
# ------------------------------------------------------------
IOT_DRIVER_COLS = list(IOT_DRIVER_COLS)
if not IOT_DRIVER_COLS:
    raise RuntimeError("[Cell9] IOT_DRIVER_COLS is empty.")

if len(IOT_DRIVER_COLS) != len(set(IOT_DRIVER_COLS)):
    raise RuntimeError("[Cell9] Duplicate IOT_DRIVER_COLS detected.")


def _drivers_raw_strict(df_part: pd.DataFrame, cols: list, split_name: str) -> tuple:
    missing_cols = [c for c in cols if c not in df_part.columns]
    if missing_cols:
        raise RuntimeError(
            f"[Cell9] {split_name}: missing required driver columns; no silent zero-fill allowed. "
            f"missing={missing_cols[:20]}"
        )

    M = df_part.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)

    bad_nonfinite = int(np.sum(~np.isfinite(M)))
    if bad_nonfinite:
        M[~np.isfinite(M)] = 0.0

    neg_count = int(np.sum(M < 0.0))
    if neg_count:
        M[M < 0.0] = 0.0

    M = np.maximum(M, 0.0).astype(np.float32, copy=False)

    return M, {
        "split": split_name,
        "shape": [int(M.shape[0]), int(M.shape[1])],
        "nonfinite_replaced_with_zero": int(bad_nonfinite),
        "negative_clipped_to_zero": int(neg_count),
        "nonzero_rate": float(np.mean(M > 0.0)) if M.size else 0.0,
        "sum_total": float(np.sum(M)) if M.size else 0.0,
    }


drv_tr, drv_tr_audit = _drivers_raw_strict(df_tr, IOT_DRIVER_COLS, "TRAIN")
drv_va, drv_va_audit = _drivers_raw_strict(df_va, IOT_DRIVER_COLS, "VAL")
drv_te_real_eval, drv_te_audit = _drivers_raw_strict(df_te, IOT_DRIVER_COLS, "TEST/EVAL_ONLY")

globals()["drv_tr"] = drv_tr
globals()["drv_va"] = drv_va
globals()["drv_te_real_eval"] = drv_te_real_eval

driver_audit = {
    "version": "cell9_v10_driver_matrix_audit",
    "driver_cols": list(IOT_DRIVER_COLS),
    "n_driver_cols": int(len(IOT_DRIVER_COLS)),
    "train": drv_tr_audit,
    "val": drv_va_audit,
    "test_eval_only": drv_te_audit,
    "policy": (
        "strict column presence; nonfinite event-driver values explicitly replaced with zero; "
        "negatives clipped to zero"
    ),
}

driver_audit_path = os.path.join(OUT_ART, "driver_matrix_audit_cell9.json")
_write_json(driver_audit_path, driver_audit)

log(f"[Cell9] Driver matrices ready: drv_tr={drv_tr.shape} | drv_va={drv_va.shape} | drv_te_real_eval={drv_te_real_eval.shape}")
log(f"[Cell9] Saved driver matrix audit: {driver_audit_path}")


# ------------------------------------------------------------
# 13) Publish runtime contract
# ------------------------------------------------------------
CELL9_MASK_DRIVER_CONTRACT = {
    "version": "cell9_v11_THESIS_mask_driver_contract",
    "tiers_full": list(TIERS_FULL),
    "tiers_ddpm": list(TIERS_DDPM),

    "syn_masks_val_full_shape": [int(SYN_MASKS_VAL_FULL.shape[0]), int(SYN_MASKS_VAL_FULL.shape[1])],
    "syn_masks_val_ddpm_shape": [int(SYN_MASKS_VAL_DDPM.shape[0]), int(SYN_MASKS_VAL_DDPM.shape[1])],
    "syn_masks_test_full_shape": [int(SYN_MASKS_TEST_FULL.shape[0]), int(SYN_MASKS_TEST_FULL.shape[1])],
    "syn_masks_test_ddpm_shape": [int(SYN_MASKS_TEST_DDPM.shape[0]), int(SYN_MASKS_TEST_DDPM.shape[1])],

    "syn_masks_full_shape": [int(SYN_MASKS_FULL.shape[0]), int(SYN_MASKS_FULL.shape[1])],
    "syn_masks_ddpm_shape": [int(SYN_MASKS_DDPM.shape[0]), int(SYN_MASKS_DDPM.shape[1])],

    "real_val_masks_full_shape": [int(REAL_VAL_MASKS_FULL.shape[0]), int(REAL_VAL_MASKS_FULL.shape[1])],
    "real_test_masks_full_shape": [int(REAL_TEST_MASKS_FULL.shape[0]), int(REAL_TEST_MASKS_FULL.shape[1])],

    "syn_val_rates_full": dict(syn_val_rates_full),
    "syn_val_rates_ddpm": dict(syn_val_rates_ddpm),
    "syn_test_rates_full": dict(syn_test_rates_full),
    "syn_test_rates_ddpm": dict(syn_test_rates_ddpm),

    "target_pi_by_tier": dict(target_pi_by_tier),

    "mask_array_npz_path": mask_npz_path,
    "tier_mask_model_path": tier_mask_model_path,
    "tier_mask_eval_diagnostics_path": tier_mask_eval_path,
    "driver_matrix_audit_path": driver_audit_path,

    "driver_cols_n": int(len(IOT_DRIVER_COLS)),
    "drv_tr_shape": [int(drv_tr.shape[0]), int(drv_tr.shape[1])],
    "drv_va_shape": [int(drv_va.shape[0]), int(drv_va.shape[1])],
    "drv_te_real_eval_shape": [int(drv_te_real_eval.shape[0]), int(drv_te_real_eval.shape[1])],

    "a0_a1_a2_contract": {
        "reference_val_mask_source": "REAL_VAL_MASKS_FULL_EVAL_OR_SELECTOR_REFERENCE_ONLY",
        "reference_test_mask_source": "REAL_TEST_MASKS_FULL_EVAL_ONLY",
        "A0_mask_source": "not_authorized_by_Cell9",
        "A1_VAL_mask_source": "SYN_MASKS_VAL_FULL",
        "A1_TEST_mask_source": "SYN_MASKS_TEST_FULL",
        "A2_TEST_mask_source": "SYN_MASKS_TEST_FULL",
        "backward_compatibility": {
            "SYN_MASKS_FULL": "alias_of_SYN_MASKS_TEST_FULL",
            "SYN_MASKS_DDPM": "alias_of_SYN_MASKS_TEST_DDPM",
        },
    },

    "test_leakage_guard": {
        "model_artifact_contains_test_real_metrics": False,
        "test_metrics_artifact_is_eval_only": True,
        "test_used_for_fitting": False,
        "test_used_for_target_calibration": False,
        "test_real_values_used_for_generation": False,
        "test_horizon_regime_used_for_synthetic_generation": True,
        "test_used_for_model_selection": False,
    },
}

globals()["CELL9_MASK_DRIVER_CONTRACT"] = CELL9_MASK_DRIVER_CONTRACT

cell9_contract_path = os.path.join(OUT_ART, "cell9_mask_driver_contract.json")
cell9_contract_path_canonical = os.path.join(OUT_CONTRACT, "cell9_mask_driver_contract_v11_THESIS.json")
_write_json(cell9_contract_path, CELL9_MASK_DRIVER_CONTRACT)
_write_json(cell9_contract_path_canonical, CELL9_MASK_DRIVER_CONTRACT)

if "RUN_META" in globals():
    RUN_META.setdefault("generator_contracts", {})
    RUN_META["generator_contracts"]["Cell9_mask_driver_contract"] = CELL9_MASK_DRIVER_CONTRACT

globals()["CELL9_MASK_DRIVER_CONTRACT_PATH"] = cell9_contract_path_canonical

log(f"[Cell9] Saved runtime contract: {cell9_contract_path}")
log(f"[Cell9] Saved canonical contract: {cell9_contract_path_canonical}")

log(
    "[Cell9] Published masks | "
    f"SYN_MASKS_VAL_FULL={SYN_MASKS_VAL_FULL.shape} | "
    f"SYN_MASKS_TEST_FULL={SYN_MASKS_TEST_FULL.shape} | "
    f"SYN_MASKS_FULL(alias TEST)={SYN_MASKS_FULL.shape}"
)

log("--- END:   Cell 9 — tier masks + drivers (v11-THESIS VAL+TEST-horizon mask synthesis) ---")