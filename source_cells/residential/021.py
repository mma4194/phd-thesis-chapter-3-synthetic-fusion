# ==========================================================
# CELL 8.4 — DDPM runtime/artifact sanity checks — v15-THESIS
# ACTIVE SeqDenoiser CHECKPOINT / SPEC / SCALER / EMA / x_obs
# + CELL 8 v18 ADVISORY-ONLY DDPM CANDIDATE REGISTRY CONTRACT
#
# Purpose:
# - Validate that freshly trained DDPM artifacts match the current
#   in-memory canonical DDPM target contract.
# - Validate Cell 8 v18 advisory-only DDPM candidate-generator contract.
# - Treat DDPM as a conditional candidate generator, not as a direct A2
#   overwrite authority.
# - Load and validate:
#       checkpoint
#       train spec
#       scaler
#       active contract
#       Cell 8 generator contract
#       x_obs audit
#       target support audit
#       target transform audit
#       advisory VAL diagnostic audit
#       DDPM candidate registry
# - Rehydrate active ddpm_net + EMA shadow from checkpoint.
# - Export downstream-safe globals for the VAL-only selector:
#       DDPM_CANDIDATE_REGISTRY
#       DDPM_VAL_DIAGNOSTIC_AUDIT
#       DDPM_ADVISORY_PASS_COLS
#       DDPM_ADVISORY_EXCLUDED_COLS
#       DDPM_DOWNSTREAM_SAFE_OVERWRITE_COLS = []
#       DDPM_DOWNSTREAM_EXCLUDED_OVERWRITE_COLS = all DDPM cols
#       DDPM_RUNTIME_AUDIT
#
# Critical policy:
# - Cell 8.4 is a runtime/artifact safety gate only.
# - Cell 8.4 does NOT select DDPM for A2.
# - Cell 8.4 does NOT treat advisory_pass=0 as failure.
# - The downstream VAL-only selector remains the only selection authority.
# - TEST is not used here.
# ==========================================================

log("--- START: Cell 8.4 — DDPM sanity checks (v15-THESIS advisory-only candidate registry) ---")

import os
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
need = [
    "CFG", "log",
    "ddpm_net",
    "DDPM_VALUE_COLS",
    "cond_tr", "cond_va",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell8.4] Missing prerequisites: {missing}. Run Cells 1–8 first.")

# ----------------------------------------------------------
# Clean-run leakage guard inherited from Cell 1
# ----------------------------------------------------------
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(
            f"[Cell8.4] Clean DDPM runtime check forbids CFG['{_flag}']=True."
        )

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell8.4] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )

ARTDIR = os.path.join(str(CFG["outdir"]), "artifacts")
REP_DIR = os.path.join(str(CFG["outdir"]), "reports")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

FAIL_ON_STALE_COND_DIM = bool(CFG.get("cell8_4_fail_on_stale_COND_DIM", True))
FAIL_ON_LEGACY_DDPM_MISMATCH = bool(CFG.get("cell8_4_fail_on_legacy_ddpm_mismatch", False))
FAIL_ON_MISSING_ADVISORY_DIAGNOSTIC = bool(CFG.get("cell8_4_fail_on_missing_ddpm_val_diagnostic", True))
FAIL_ON_MISSING_CANDIDATE_REGISTRY = bool(CFG.get("cell8_4_fail_on_missing_ddpm_candidate_registry", True))
FAIL_IF_ADVISORY_PASS_ZERO = bool(CFG.get("cell8_4_fail_if_ddpm_advisory_pass_zero", False))

EXPECTED_TARGET_MODE = "absolute_protocol_aware"
EXPECTED_DDPM_ROLE = "conditional_refinement_candidate_generator"
EXPECTED_SELECTION_AUTHORITY = "downstream_VAL_selector_only"

ACCEPTABLE_XOBS_BASIS = {
    "tier_obs_present_AND_value_isfinite",
    "protocol_tier_obs_present_AND_value_isfinite",
    "Cell6_logical_value_finite",
    "cell6_logical_value_finite",
    "Cell6_logical_tier_obs_AND_value_isfinite",
    "cell6_logical_tier_obs_AND_value_isfinite",
    "cell6_logical_tier_obs_and_value_isfinite",
}

ACCEPTABLE_XOBS_POLICY_TEXT = {
    "x_obs = protocol_tier_obs_present AND isfinite(raw_target_value)",
    "x_obs = tier_obs_present AND isfinite(raw_target_value)",
    "x_obs = Cell6 logical tier obs AND isfinite(raw_target_value)",
    "x_obs = Cell6 logical tier observability AND isfinite(raw_target_value)",
    "x_obs = Cell6 logical tier obs AND isfinite(raw target value)",
    "x_obs = Cell6 logical DDPM tier obs mask AND isfinite(raw_target_value)",
    "x_obs = Cell6 logical DDPM tier observability mask AND isfinite(raw_target_value)",
    "x_obs = Cell6 DDPM logical tier obs mask AND isfinite(raw_target_value)",
}


# ----------------------------------------------------------
# 1) Helpers
# ----------------------------------------------------------
def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
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
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json_dict(path: str, *, required: bool = True, label: str = "json") -> dict:
    if not os.path.exists(path):
        if required:
            raise RuntimeError(f"[Cell8.4] Missing {label}: {path}")
        return {"missing": True, "path": path}

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell8.4] Expected dict in {label}: {path}, got={type(obj)}")

    return obj


def _sha1_list(xs) -> str:
    h = hashlib.sha1()
    for x in xs:
        h.update((str(x) + "\n").encode("utf-8"))
    return h.hexdigest()


def _sha256_pipejoin(xs) -> str:
    return hashlib.sha256(("||".join(map(str, xs))).encode("utf-8")).hexdigest()


def _check_unique_list(name: str, xs: list) -> None:
    xs = list(xs)
    if len(xs) != len(set(xs)):
        vals, cnts = np.unique(np.asarray(xs, dtype=object), return_counts=True)
        dupes = [str(v) for v, c in zip(vals, cnts) if c > 1]
        raise RuntimeError(f"[Cell8.4] Duplicate entries in {name}: {dupes[:20]}")


def _as_list_str(obj, name: str) -> list:
    if obj is None:
        return []
    if not isinstance(obj, list):
        raise RuntimeError(f"[Cell8.4] {name} must be list, got={type(obj)}")
    return [str(x) for x in obj]


def _as_dict(obj, name: str) -> dict:
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell8.4] {name} must be dict, got={type(obj)}")
    return obj


def _validate_col_subset(name: str, subset: list, universe: list) -> None:
    bad = sorted(set(map(str, subset)) - set(map(str, universe)))
    if bad:
        raise RuntimeError(f"[Cell8.4] {name} contains columns outside DDPM_VALUE_COLS: {bad[:20]}")


def _optional_str(container: dict, key: str, default: str = "") -> str:
    return str(container.get(key, default) or "").strip()


def _first_existing_path(candidates: list, *, label: str, required: bool) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    if required:
        raise RuntimeError(
            f"[Cell8.4] Missing {label}. Tried:\n" + "\n".join(str(p) for p in candidates)
        )
    return candidates[0] if candidates else ""


def _infer_cond_dim_from_model(model):
    if model is None:
        return None

    if hasattr(model, "cond_dim"):
        try:
            return int(model.cond_dim)
        except Exception:
            pass

    cp = getattr(model, "cond_proj", None)
    if cp is not None:
        if isinstance(cp, nn.Linear):
            return int(cp.in_features)
        if isinstance(cp, nn.Sequential):
            for m in cp.modules():
                if isinstance(m, nn.Linear):
                    return int(m.in_features)

    try:
        for name, m in model.named_modules():
            if "cond" in str(name).lower() and isinstance(m, nn.Linear):
                return int(m.in_features)
    except Exception:
        pass

    return None


def _tensor_dict_is_finite(sd: dict, label: str, max_report: int = 5) -> None:
    bad = []
    for k, v in sd.items():
        if not torch.is_tensor(v):
            bad.append((k, "non_tensor"))
            continue
        if torch.is_floating_point(v) or torch.is_complex(v):
            if not torch.isfinite(v).all().item():
                bad.append((k, "non_finite"))
    if bad:
        raise RuntimeError(f"[Cell8.4] {label} invalid entries: {bad[:max_report]}")


def _check_tensor_state_keys_match(a: dict, b: dict, label_a: str, label_b: str) -> None:
    ka, kb = set(a.keys()), set(b.keys())
    missing_b = sorted(ka - kb)
    missing_a = sorted(kb - ka)
    if missing_b or missing_a:
        raise RuntimeError(
            f"[Cell8.4] State key mismatch between {label_a} and {label_b}: "
            f"missing_in_{label_b}={missing_b[:10]} | missing_in_{label_a}={missing_a[:10]}"
        )


def _shape_of_state(sd: dict, key: str):
    if key not in sd:
        raise RuntimeError(f"[Cell8.4] Missing state key: {key}")
    if not hasattr(sd[key], "shape"):
        raise RuntimeError(f"[Cell8.4] State key is not tensor-like: {key}")
    return tuple(sd[key].shape)


def _rehydrate_ema_shadow(active_wrapper, ema_shadow_ckpt: dict) -> None:
    if active_wrapper is None:
        raise RuntimeError("[Cell8.4] Cannot rehydrate EMA: active wrapper is None.")
    if not hasattr(active_wrapper, "ema") or active_wrapper.ema is None:
        raise RuntimeError("[Cell8.4] Active wrapper has no EMA object.")
    if not hasattr(active_wrapper.ema, "shadow"):
        raise RuntimeError("[Cell8.4] Active EMA has no shadow attribute.")

    _check_tensor_state_keys_match(
        active_wrapper.ema.shadow,
        ema_shadow_ckpt,
        "active ddpm_seq.ema.shadow",
        "checkpoint ema_shadow",
    )

    new_shadow = {}
    for k, v in ema_shadow_ckpt.items():
        if not torch.is_tensor(v):
            raise RuntimeError(f"[Cell8.4] EMA checkpoint entry is not tensor: {k}")

        target = active_wrapper.ema.shadow[k]
        vv = v.detach().clone().contiguous()
        if vv.device != target.device:
            vv = vv.to(target.device)
        if vv.dtype != target.dtype:
            vv = vv.to(dtype=target.dtype)
        new_shadow[k] = vv

    active_wrapper.ema.shadow = new_shadow


def _canonical_torch_device_str(dev) -> str:
    if dev is None:
        return "unknown"

    s = str(dev).strip().lower()

    if s == "cuda":
        if torch.cuda.is_available():
            return f"cuda:{int(torch.cuda.current_device())}"
        return "cuda"

    if s.startswith("cuda:"):
        return s

    if s == "cpu":
        return "cpu"

    return s


def _resolve_device_for_ckpt_load() -> str:
    validate_on_cpu = bool(CFG.get("cell8_4_validate_ckpt_on_cpu", False))
    if validate_on_cpu:
        return "cpu"

    req = str(CFG.get("device", "auto")).lower().strip()

    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return req
        log(f"[Cell8.4] Requested device={req!r}, but CUDA is unavailable. Loading on CPU.")
        return "cpu"

    return "cpu"


def _container_any(container: dict, keys: list, default=None):
    for k in keys:
        if isinstance(container, dict) and k in container:
            return container.get(k)
    return default


def _require_target_mode(container: dict, label: str) -> None:
    mode = _container_any(container, ["target_mode", "ddpm_target_mode"], "")
    mode = str(mode or "").strip()
    if mode and mode != EXPECTED_TARGET_MODE:
        raise RuntimeError(
            f"[Cell8.4] {label} target_mode mismatch: got={mode!r}, expected={EXPECTED_TARGET_MODE!r}"
        )


def _require_role_compatible(container: dict, label: str) -> None:
    role = _container_any(container, ["ddpm_role", "role", "model_role"], "")
    role = str(role or "").strip()

    if role and role != EXPECTED_DDPM_ROLE:
        raise RuntimeError(
            f"[Cell8.4] {label} DDPM role mismatch: got={role!r}, expected={EXPECTED_DDPM_ROLE!r}"
        )


def _require_xobs_basis_compatible(container: dict, label: str) -> None:
    basis = _container_any(container, ["x_obs_basis", "xobs_basis", "x_obs"], "")
    basis = str(basis or "").strip()

    if basis and basis not in ACCEPTABLE_XOBS_BASIS:
        raise RuntimeError(
            f"[Cell8.4] {label} x_obs_basis mismatch: got={basis!r}, "
            f"allowed={sorted(ACCEPTABLE_XOBS_BASIS)}"
        )


def _extract_rows(container: dict) -> list:
    for k in ["rows_best", "rows", "audit_rows", "diagnostic_rows", "screen_rows", "val_screen_rows"]:
        rows = container.get(k, None)
        if rows is not None:
            if not isinstance(rows, list):
                raise RuntimeError(f"[Cell8.4] {k} must be list, got={type(rows)}")
            return rows
    return []


def _extract_advisory_pass_cols(diag: dict, cols_now: list) -> list:
    keys = [
        "advisory_pass_cols",
        "advisory_pass_columns",
        "ddpm_advisory_pass_cols",
        "safe_overwrite_cols",
        "cell11_safe_overwrite_cols",
        "safe_for_Cell11_cols",
        "safe_for_cell11_cols",
        "accepted_cols",
    ]

    for k in keys:
        if k in diag:
            cols = _as_list_str(diag.get(k), f"diagnostic.{k}")
            _validate_col_subset(f"diagnostic.{k}", cols, cols_now)
            _check_unique_list(f"diagnostic.{k}", cols)
            return cols

    rows = _extract_rows(diag)
    out = []

    for r in rows:
        if not isinstance(r, dict):
            continue

        col = r.get("col", None)
        if col is None:
            continue

        decision = str(r.get("decision", r.get("status", ""))).strip().lower()
        advisory = r.get("advisory_pass", r.get("pass", r.get("safe_for_Cell11", None)))

        ok = (
            decision in {"accept", "accepted", "pass", "passed", "advisory_pass"}
            or advisory is True
            or advisory == 1
            or str(advisory).strip().lower() in {"true", "1", "pass", "passed"}
        )

        if ok:
            out.append(str(col))

    _validate_col_subset("diagnostic advisory-pass rows", out, cols_now)
    _check_unique_list("diagnostic advisory-pass rows", out)
    return out


def _extract_best_candidate_by_col(container: dict, cols_now: list) -> dict:
    obj = _container_any(
        container,
        ["best_candidate_by_col", "best_diagnostic_candidate_by_col", "recommendation_by_col"],
        None,
    )

    if isinstance(obj, dict):
        out = {str(k): v for k, v in obj.items() if str(k) in set(cols_now)}
        if set(out.keys()) == set(cols_now):
            return out

    rows = _extract_rows(container)
    out = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        col = r.get("col", None)
        if col is None:
            continue
        col = str(col)
        if col in set(cols_now):
            out[col] = dict(r)

    missing = sorted(set(cols_now) - set(out.keys()))
    if missing:
        # Advisory files may summarize rather than row-expand. Return placeholder rows
        # instead of failing, because Cell 11 selection should use the candidate registry.
        out = {
            str(c): {
                "col": str(c),
                "decision": "not_available_in_advisory_rows",
                "reason": "candidate_registry_is_authoritative_for_generation",
            }
            for c in cols_now
        }

    return out


def _extract_decision_counts(container: dict) -> dict:
    for k in ["decision_counts", "advisory_decision_counts"]:
        if isinstance(container.get(k, None), dict):
            return {str(a): int(b) for a, b in container[k].items()}

    rows = _extract_rows(container)
    out = {}
    for r in rows:
        if isinstance(r, dict):
            d = str(r.get("decision", r.get("status", "unknown"))).strip().lower()
            out[d] = out.get(d, 0) + 1
    return out


def _extract_failure_counts(container: dict) -> dict:
    for k in ["failure_reason_counts", "top_failures", "advisory_failure_reason_counts"]:
        if isinstance(container.get(k, None), dict):
            return {str(a): int(b) for a, b in container[k].items()}

    rows = _extract_rows(container)
    out = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        fr = r.get("failure_reasons", [])
        if isinstance(fr, str):
            fr = [fr]
        if not isinstance(fr, list):
            fr = []
        for x in fr:
            out[str(x)] = out.get(str(x), 0) + 1

    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def _normalize_registry_cols(registry: dict, cols_now: list) -> dict:
    """
    Validate the DDPM candidate registry without assuming exact old/new key names.
    The registry is authoritative for downstream candidate availability.
    """
    if not isinstance(registry, dict):
        raise RuntimeError("[Cell8.4] DDPM candidate registry must be dict.")

    reg_cols = _container_any(
        registry,
        ["ddpm_cols_use", "DDPM_VALUE_COLS", "cols", "candidate_cols", "ddpm_candidate_cols"],
        None,
    )

    if reg_cols is not None:
        reg_cols = _as_list_str(reg_cols, "candidate_registry.cols")
        if reg_cols != list(cols_now):
            mismatch_i = next((i for i, (a, b) in enumerate(zip(reg_cols, cols_now)) if a != b), None)
            raise RuntimeError(
                "[Cell8.4] Candidate registry DDPM column order mismatch.\n"
                f"first_mismatch={mismatch_i}\n"
                f"registry_col={reg_cols[mismatch_i] if mismatch_i is not None else 'NA'}\n"
                f"current_col={cols_now[mismatch_i] if mismatch_i is not None else 'NA'}"
            )

    families = _container_any(
        registry,
        ["candidate_families", "ddpm_candidate_families", "families"],
        [],
    )
    families = _as_list_str(families, "candidate_registry.candidate_families")

    if not families:
        # Fall back to global list if Cell 8 exported it.
        families = _as_list_str(globals().get("DDPM_CANDIDATE_FAMILIES", []), "DDPM_CANDIDATE_FAMILIES")

    if not families:
        raise RuntimeError("[Cell8.4] Candidate registry has no candidate families.")

    required_core = {"ddpm_absolute"}
    missing_core = sorted(required_core - set(families))
    if missing_core:
        raise RuntimeError(f"[Cell8.4] Candidate registry missing core DDPM family: {missing_core}")

    registry["candidate_families_normalized"] = list(families)
    registry["ddpm_cols_use_normalized"] = list(cols_now)
    registry["D_normalized"] = int(len(cols_now))

    return registry


# ----------------------------------------------------------
# 2) Current DDPM value contract
# ----------------------------------------------------------
cols_now = list(DDPM_VALUE_COLS)
D_now = int(len(cols_now))

if D_now <= 0:
    raise RuntimeError("[Cell8.4] DDPM_VALUE_COLS is empty.")

_check_unique_list("DDPM_VALUE_COLS", cols_now)

cols_sig_sha1_now = _sha1_list(cols_now)
cols_sig_sha256_now = _sha256_pipejoin(cols_now)

ddpm_sig_global = str(globals().get("DDPM_VALUE_SIG", "") or "").strip()
if ddpm_sig_global and ddpm_sig_global != cols_sig_sha256_now:
    raise RuntimeError(
        "[Cell8.4] DDPM_VALUE_SIG mismatch vs active DDPM_VALUE_COLS: "
        f"global={ddpm_sig_global[:12]} active={cols_sig_sha256_now[:12]}"
    )

if "PROTO_COLS_CONTRACT" in globals():
    proto_cols = list(globals()["PROTO_COLS_CONTRACT"])
    bad = sorted(set(cols_now) - set(proto_cols))
    if bad:
        raise RuntimeError(f"[Cell8.4] DDPM_VALUE_COLS not subset of PROTO_COLS_CONTRACT: {bad[:20]}")

log(
    f"[Cell8.4] Current DDPM value contract: "
    f"D={D_now} | sha1={cols_sig_sha1_now[:12]} | sha256={cols_sig_sha256_now[:12]}"
)


# ----------------------------------------------------------
# 3) Current conditioning contract
# ----------------------------------------------------------
cond_tr_arr = np.asarray(cond_tr)
cond_va_arr = np.asarray(cond_va)

if cond_tr_arr.ndim != 2:
    raise RuntimeError(f"[Cell8.4] cond_tr must be 2D, got={cond_tr_arr.shape}")
if cond_va_arr.ndim != 2:
    raise RuntimeError(f"[Cell8.4] cond_va must be 2D, got={cond_va_arr.shape}")
if cond_tr_arr.shape[1] != cond_va_arr.shape[1]:
    raise RuntimeError(
        f"[Cell8.4] cond_tr/cond_va dim mismatch: "
        f"tr={cond_tr_arr.shape[1]} va={cond_va_arr.shape[1]}"
    )
if not np.isfinite(cond_tr_arr).all():
    raise RuntimeError("[Cell8.4] cond_tr contains non-finite values.")
if not np.isfinite(cond_va_arr).all():
    raise RuntimeError("[Cell8.4] cond_va contains non-finite values.")

cond_dim_now = int(cond_tr_arr.shape[1])

if "cond_te" in globals():
    cond_te_arr = np.asarray(globals()["cond_te"])
    if cond_te_arr.ndim != 2:
        raise RuntimeError(f"[Cell8.4] cond_te must be 2D if present, got={cond_te_arr.shape}")
    if cond_te_arr.shape[1] != cond_dim_now:
        raise RuntimeError(
            f"[Cell8.4] cond_te dim mismatch: te={cond_te_arr.shape[1]} tr={cond_dim_now}"
        )
    if not np.isfinite(cond_te_arr).all():
        raise RuntimeError("[Cell8.4] cond_te contains non-finite values.")

cond_spec_path = os.path.join(ARTDIR, "cond_spec.json")
cond_spec = _read_json_dict(cond_spec_path, required=True, label="cond_spec.json")

cond_dim_spec = int(cond_spec.get("cond_dim", -1))
if cond_dim_spec <= 0:
    raise RuntimeError("[Cell8.4] cond_spec.json missing/invalid cond_dim.")

if cond_dim_now != cond_dim_spec:
    raise RuntimeError(
        f"[Cell8.4] Active cond_dim mismatch vs cond_spec: active={cond_dim_now} spec={cond_dim_spec}"
    )

cond_sig_spec = str(cond_spec.get("sig_cond", "") or "").strip()
cond_ver_spec = str(cond_spec.get("version", "") or "").strip()

if "COND_DIM" in globals() and int(globals()["COND_DIM"]) != cond_dim_now:
    msg = (
        f"[Cell8.4] Stale COND_DIM: global={globals()['COND_DIM']} active={cond_dim_now}."
    )
    if FAIL_ON_STALE_COND_DIM:
        raise RuntimeError(msg)
    log("[Cell8.4][FIX] " + msg + " Updating global COND_DIM.")
    globals()["COND_DIM"] = int(cond_dim_now)
else:
    globals()["COND_DIM"] = int(cond_dim_now)

log(
    f"[Cell8.4] Current conditioning contract: cond_dim={cond_dim_now}"
    + (f" | cond_sig={cond_sig_spec[:12]}" if cond_sig_spec else "")
    + (f" | version={cond_ver_spec}" if cond_ver_spec else "")
)


# ----------------------------------------------------------
# 4) Resolve artifact paths
# ----------------------------------------------------------
sig12 = str(globals().get("ddpm_sig12", "") or "").strip()
if not sig12:
    sig12 = cols_sig_sha256_now[:12]
    globals()["ddpm_sig12"] = sig12
    log(f"[Cell8.4][FIX] Missing ddpm_sig12; derived sig12={sig12}")

ckpt_path = os.path.join(ARTDIR, f"ddpm_core_ckpt_sig{sig12}_D{D_now}.pt")
spec_path = os.path.join(ARTDIR, f"ddpm_train_spec_sig{sig12}_D{D_now}.json")
scaler_path = os.path.join(ARTDIR, f"ddpm_scaler_sig{sig12}_D{D_now}.joblib")
xobs_audit_path = os.path.join(ARTDIR, f"ddpm_xobs_value_finite_audit_sig{sig12}_D{D_now}.json")
target_support_path = os.path.join(ARTDIR, f"ddpm_target_support_audit_sig{sig12}_D{D_now}.json")
target_transform_path = os.path.join(ARTDIR, f"ddpm_target_transform_audit_sig{sig12}_D{D_now}.json")
active_contract_path = os.path.join(ARTDIR, "ddpm_active_contract_from_cell8.json")
cell8_generator_contract_path = os.path.join(
    CONTRACT_DIR,
    "ddpm_cell8_generator_contract.json",
)

# Cell 8 v17 canonical names.
val_diag_path = _first_existing_path(
    [
        os.path.join(ARTDIR, f"ddpm_val_diagnostic_audit_sig{sig12}_D{D_now}.json"),
        os.path.join(ARTDIR, f"ddpm_val_screen_audit_sig{sig12}_D{D_now}.json"),
    ],
    label="DDPM VAL diagnostic/screen audit",
    required=FAIL_ON_MISSING_ADVISORY_DIAGNOSTIC,
)

candidate_registry_path = _first_existing_path(
    [
        os.path.join(ARTDIR, f"ddpm_candidate_registry_sig{sig12}_D{D_now}.json"),
        os.path.join(ARTDIR, f"ddpm_refinement_recommendation_sig{sig12}_D{D_now}.json"),
    ],
    label="DDPM candidate registry",
    required=FAIL_ON_MISSING_CANDIDATE_REGISTRY,
)

required_paths = [
    (ckpt_path, "checkpoint"),
    (spec_path, "train spec"),
    (scaler_path, "target scaler"),
    (xobs_audit_path, "x_obs audit"),
    (target_support_path, "target support audit"),
    (target_transform_path, "target transform audit"),
    (active_contract_path, "active DDPM contract"),
    (cell8_generator_contract_path, "Cell 8 generator contract"),
]

for p, label in required_paths:
    if not os.path.exists(p):
        raise RuntimeError(f"[Cell8.4] Missing {label}: {p}")


# ----------------------------------------------------------
# 5) Load artifacts
# ----------------------------------------------------------
map_location = _resolve_device_for_ckpt_load()

try:
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
except TypeError:
    ckpt = torch.load(ckpt_path, map_location=map_location)

if not isinstance(ckpt, dict):
    raise RuntimeError(f"[Cell8.4] Checkpoint must be dict: {ckpt_path}")

ckpt_spec = ckpt.get("spec", None)
if not isinstance(ckpt_spec, dict):
    raise RuntimeError("[Cell8.4] Checkpoint missing dict ckpt['spec'].")

spec_json = _read_json_dict(spec_path, required=True, label="train spec")
xobs_audit = _read_json_dict(xobs_audit_path, required=True, label="x_obs audit")
target_support_audit = _read_json_dict(target_support_path, required=True, label="target support audit")
target_transform_audit = _read_json_dict(target_transform_path, required=True, label="target transform audit")
active_contract = _read_json_dict(active_contract_path, required=True, label="active contract")
cell8_generator_contract = _read_json_dict(
    cell8_generator_contract_path,
    required=True,
    label="Cell 8 generator contract",
)
ddpm_scaler_loaded = joblib.load(scaler_path)

val_diag_audit = _read_json_dict(
    val_diag_path,
    required=FAIL_ON_MISSING_ADVISORY_DIAGNOSTIC,
    label="VAL diagnostic audit",
)

candidate_registry = _read_json_dict(
    candidate_registry_path,
    required=FAIL_ON_MISSING_CANDIDATE_REGISTRY,
    label="candidate registry",
)

log(f"[Cell8.4] Loaded checkpoint: {os.path.basename(ckpt_path)} | map_location={map_location}")
log(f"[Cell8.4] Loaded train spec: {os.path.basename(spec_path)}")
log(f"[Cell8.4] Loaded scaler: {os.path.basename(scaler_path)}")
log(f"[Cell8.4] Loaded active contract: {os.path.basename(active_contract_path)}")
log(f"[Cell8.4] Loaded Cell 8 generator contract: {os.path.basename(cell8_generator_contract_path)}")
log(f"[Cell8.4] Loaded VAL diagnostic: {os.path.basename(val_diag_path)}")
log(f"[Cell8.4] Loaded candidate registry: {os.path.basename(candidate_registry_path)}")


# ----------------------------------------------------------
# 6) DDPM column/signature contract checks
# ----------------------------------------------------------
for label, container in [
    ("checkpoint spec", ckpt_spec),
    ("checkpoint top-level", ckpt),
    ("train spec", spec_json),
    ("active contract", active_contract),
]:
    cols = container.get("ddpm_cols_use", None)
    D_val = container.get("D", None)

    if cols is not None:
        if not isinstance(cols, list):
            raise RuntimeError(f"[Cell8.4] {label} ddpm_cols_use must be list.")
        if list(cols) != cols_now:
            mismatch_i = next((i for i, (a, b) in enumerate(zip(cols, cols_now)) if a != b), None)
            raise RuntimeError(
                f"[Cell8.4] {label} ddpm_cols_use mismatch.\n"
                f"first_mismatch={mismatch_i}\n"
                f"{label}_col={cols[mismatch_i] if mismatch_i is not None else 'NA'}\n"
                f"active_col={cols_now[mismatch_i] if mismatch_i is not None else 'NA'}"
            )

    if D_val is not None and int(D_val) != D_now:
        raise RuntimeError(f"[Cell8.4] {label} D mismatch: {D_val} vs active={D_now}")

    sig_val = str(container.get("sig12", container.get("ddpm_sig12", "")) or "").strip()
    if sig_val and sig_val != sig12:
        raise RuntimeError(f"[Cell8.4] {label} sig12 mismatch: {sig_val} vs active={sig12}")

log(f"[Cell8.4] OK: checkpoint/spec/current DDPM target contract match | sig12={sig12}")


# ----------------------------------------------------------
# 7) Conditioning contract checks
# ----------------------------------------------------------
for label, container in [
    ("checkpoint spec", ckpt_spec),
    ("checkpoint top-level", ckpt),
    ("train spec", spec_json),
    ("active contract", active_contract),
]:
    cdim = container.get("cond_dim", None)
    if cdim is not None and int(cdim) != cond_dim_now:
        raise RuntimeError(f"[Cell8.4] {label} cond_dim mismatch: {cdim} vs active={cond_dim_now}")

    csig = str(container.get("cond_sig", "") or "").strip()
    if cond_sig_spec and csig and csig != cond_sig_spec:
        raise RuntimeError(
            f"[Cell8.4] {label} cond_sig mismatch: {csig[:12]} vs cond_spec={cond_sig_spec[:12]}"
        )

    cver = str(container.get("cond_spec_version", "") or "").strip()
    if cond_ver_spec and cver and cver != cond_ver_spec:
        raise RuntimeError(
            f"[Cell8.4] {label} cond_spec_version mismatch: {cver} vs {cond_ver_spec}"
        )

log("[Cell8.4] OK: conditioning contract matches current tensors/spec/checkpoint")


# ----------------------------------------------------------
# 8) Role / target mode / x_obs checks
# ----------------------------------------------------------
for label, container in [
    ("checkpoint spec", ckpt_spec),
    ("checkpoint top-level", ckpt),
    ("train spec", spec_json),
    ("active contract", active_contract),
]:
    _require_target_mode(container, label)
    _require_role_compatible(container, label)
    _require_xobs_basis_compatible(container, label)

selection_authority = str(
    active_contract.get(
        "selection_authority",
        spec_json.get("selection_authority", EXPECTED_SELECTION_AUTHORITY),
    )
    or EXPECTED_SELECTION_AUTHORITY
).strip()

legacy_direct_overwrite_disabled = bool(
    active_contract.get(
        "legacy_direct_overwrite_disabled",
        spec_json.get("legacy_direct_overwrite_disabled", True),
    )
)

if selection_authority != EXPECTED_SELECTION_AUTHORITY:
    raise RuntimeError(
        f"[Cell8.4] DDPM selection_authority mismatch: got={selection_authority!r}, "
        f"expected={EXPECTED_SELECTION_AUTHORITY!r}"
    )

if not legacy_direct_overwrite_disabled:
    raise RuntimeError(
        "[Cell8.4] legacy_direct_overwrite_disabled is False. "
        "Cell 8 v17 must not authorize direct DDPM overwrite."
    )

def _normalize_policy_text(s: str) -> str:
    return (
        str(s or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

def _is_compatible_xobs_policy_text(s: str) -> bool:
    z = _normalize_policy_text(s)

    has_xobs = "xobs=" in z or z.startswith("xobs")
    has_cell6_or_protocol_obs = (
        "cell6" in z
        or "protocoltierobspresent" in z
        or "tierobspresent" in z
        or "logicalddpmtierobsmask" in z
        or "logicaltierobs" in z
    )
    has_finite_raw_value = (
        "isfinite(rawtargetvalue)" in z
        or "isfinite(rawvalue)" in z
        or "valueisfinite" in z
    )
    has_and = "and" in z

    return bool(has_xobs and has_cell6_or_protocol_obs and has_finite_raw_value and has_and)

audit_policy = str(xobs_audit.get("policy", "") or "").strip()
if not _is_compatible_xobs_policy_text(audit_policy):
    raise RuntimeError(
        f"[Cell8.4] x_obs audit policy mismatch: got={audit_policy!r}. "
        "Expected semantic policy: x_obs = Cell6/protocol logical tier observability AND finite raw target value."
    )

for split, expected_rows in [("train", cond_tr_arr.shape[0]), ("val", cond_va_arr.shape[0])]:
    if split not in xobs_audit:
        raise RuntimeError(f"[Cell8.4] x_obs audit missing split: {split}")

    row = xobs_audit[split]
    if not isinstance(row, dict):
        raise RuntimeError(f"[Cell8.4] x_obs audit {split} must be dict.")

    shape = row.get("shape", None)
    if not isinstance(shape, list) or len(shape) != 2:
        raise RuntimeError(f"[Cell8.4] x_obs audit {split}.shape invalid: {shape}")

    if int(shape[0]) != int(expected_rows) or int(shape[1]) != D_now:
        raise RuntimeError(
            f"[Cell8.4] x_obs audit {split}.shape mismatch: "
            f"got={shape}, expected={[int(expected_rows), D_now]}"
        )

    final_rate = float(row.get("final_x_obs_rate", -1.0))
    finite_rate = float(row.get("value_finite_rate", -1.0))
    tier_rate = float(row.get("tier_only_obs_rate", -1.0))

    if not (0.0 <= final_rate <= 1.0 and 0.0 <= finite_rate <= 1.0 and 0.0 <= tier_rate <= 1.0):
        raise RuntimeError(f"[Cell8.4] x_obs audit {split} rates outside [0,1]: {row}")

    if final_rate > min(finite_rate, tier_rate) + 1e-6:
        raise RuntimeError(
            f"[Cell8.4] x_obs audit {split} violates AND policy: "
            f"final={final_rate}, finite={finite_rate}, tier={tier_rate}"
        )

log(
    "[Cell8.4] OK: role/target/x_obs contract validated | "
    f"selection_authority={selection_authority} | legacy_direct_overwrite_disabled={legacy_direct_overwrite_disabled} | "
    f"xobs_train={float(xobs_audit['train']['final_x_obs_rate']):.6f} | "
    f"xobs_val={float(xobs_audit['val']['final_x_obs_rate']):.6f}"
)

# ----------------------------------------------------------
# 8b) Cell 8 generator-contract validation
# ----------------------------------------------------------
if str(cell8_generator_contract.get("generator_family", "")) != "DDPM":
    raise RuntimeError(
        "[Cell8.4] Cell 8 generator contract generator_family mismatch: "
        f"{cell8_generator_contract.get('generator_family')!r}"
    )

if bool(cell8_generator_contract.get("contributes_to_final_artifact", True)):
    raise RuntimeError(
        "[Cell8.4] Cell 8 generator contract incorrectly marks DDPM as final-contributing."
    )

leak = cell8_generator_contract.get("leakage_status", {})
if not isinstance(leak, dict):
    raise RuntimeError("[Cell8.4] Cell 8 generator contract missing leakage_status dict.")

for _k in [
    "uses_test_for_selection",
    "uses_test_for_repair",
    "mutates_synthetic_artifact",
    "overwrites_final_artifact",
]:
    if bool(leak.get(_k, False)):
        raise RuntimeError(
            f"[Cell8.4] Cell 8 generator contract violates leakage rule: {_k}=True."
        )

log("[Cell8.4] OK: Cell 8 generator contract validated as candidate-only, non-final-contributing.")


# ----------------------------------------------------------
# 9) Scaler checks
# ----------------------------------------------------------
if not (hasattr(ddpm_scaler_loaded, "transform") and hasattr(ddpm_scaler_loaded, "inverse_transform")):
    raise RuntimeError("[Cell8.4] Loaded scaler is not transform/inverse_transform compatible.")

n_features = getattr(ddpm_scaler_loaded, "n_features_in_", None)
if not isinstance(n_features, (int, np.integer)):
    raise RuntimeError("[Cell8.4] Loaded scaler missing n_features_in_.")

if int(n_features) != D_now:
    raise RuntimeError(f"[Cell8.4] Scaler n_features mismatch: {int(n_features)} vs D={D_now}")

if hasattr(ddpm_scaler_loaded, "center_"):
    center = np.asarray(ddpm_scaler_loaded.center_)
    if center.shape[0] != D_now or not np.isfinite(center).all():
        raise RuntimeError("[Cell8.4] Scaler center_ invalid.")

if hasattr(ddpm_scaler_loaded, "scale_"):
    scale = np.asarray(ddpm_scaler_loaded.scale_)
    if scale.shape[0] != D_now or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError("[Cell8.4] Scaler scale_ invalid.")

if hasattr(ddpm_scaler_loaded, "transform_kind_"):
    kinds = list(ddpm_scaler_loaded.transform_kind_)
    if len(kinds) != D_now:
        raise RuntimeError(
            f"[Cell8.4] Scaler transform_kind_ length mismatch: {len(kinds)} vs D={D_now}"
        )

globals()["ddpm_scaler"] = ddpm_scaler_loaded

log(f"[Cell8.4] OK: scaler validated | type={type(ddpm_scaler_loaded).__name__} | D={D_now}")


# ----------------------------------------------------------
# 10) Checkpoint weights and EMA checks
# ----------------------------------------------------------
state_dict = ckpt.get("model_state_dict", None)
ema_shadow_ckpt = ckpt.get("ema_shadow", None)

if not isinstance(state_dict, dict) or not state_dict:
    raise RuntimeError("[Cell8.4] checkpoint model_state_dict missing/empty.")
if not isinstance(ema_shadow_ckpt, dict) or not ema_shadow_ckpt:
    raise RuntimeError("[Cell8.4] checkpoint ema_shadow missing/empty.")

_tensor_dict_is_finite(state_dict, "checkpoint model_state_dict")
_tensor_dict_is_finite(ema_shadow_ckpt, "checkpoint ema_shadow")
_check_tensor_state_keys_match(state_dict, ema_shadow_ckpt, "model_state_dict", "ema_shadow")

log(f"[Cell8.4] OK: checkpoint tensors finite | model_tensors={len(state_dict)} | ema_tensors={len(ema_shadow_ckpt)}")


# ----------------------------------------------------------
# 11) Active runtime alias validation
# ----------------------------------------------------------
if type(globals()["ddpm_net"]).__name__ != "SeqDenoiser":
    raise RuntimeError(
        f"[Cell8.4] Active ddpm_net must be SeqDenoiser, got={type(globals()['ddpm_net']).__name__}"
    )

active_wrapper = globals().get("ddpm_seq", None)
compat_wrapper = globals().get("ddpm", None)
compat_model = globals().get("ddpm_model", None)

if active_wrapper is None:
    raise RuntimeError("[Cell8.4] Missing active wrapper global ddpm_seq.")

if type(active_wrapper).__name__ != "DDPM":
    raise RuntimeError(f"[Cell8.4] ddpm_seq must be DDPM wrapper, got={type(active_wrapper).__name__}")

if compat_wrapper is not active_wrapper:
    msg = "[Cell8.4] ddpm compatibility alias is not ddpm_seq."
    if FAIL_ON_LEGACY_DDPM_MISMATCH:
        raise RuntimeError(msg)
    log("[Cell8.4][FIX] " + msg + " Updating globals()['ddpm'].")
    globals()["ddpm"] = active_wrapper
    compat_wrapper = active_wrapper

if compat_model is not globals()["ddpm_net"]:
    msg = "[Cell8.4] ddpm_model compatibility alias is not ddpm_net."
    if FAIL_ON_LEGACY_DDPM_MISMATCH:
        raise RuntimeError(msg)
    log("[Cell8.4][FIX] " + msg + " Updating globals()['ddpm_model'].")
    globals()["ddpm_model"] = globals()["ddpm_net"]
    compat_model = globals()["ddpm_net"]

if not hasattr(active_wrapper, "net") or active_wrapper.net is not globals()["ddpm_net"]:
    msg = "[Cell8.4] ddpm_seq.net is not the active ddpm_net."
    if FAIL_ON_LEGACY_DDPM_MISMATCH:
        raise RuntimeError(msg)
    log("[Cell8.4][FIX] " + msg + " Updating active_wrapper.net.")
    active_wrapper.net = globals()["ddpm_net"]


# ----------------------------------------------------------
# 12) Rehydrate active runtime from checkpoint
# ----------------------------------------------------------
net_sd = globals()["ddpm_net"].state_dict()

for key in ["in_proj.weight", "out_proj.weight"]:
    if key not in state_dict:
        raise RuntimeError(f"[Cell8.4] checkpoint missing SeqDenoiser key: {key}")
    if key not in net_sd:
        raise RuntimeError(f"[Cell8.4] active ddpm_net missing SeqDenoiser key: {key}")

ckpt_in_proj_shape = _shape_of_state(state_dict, "in_proj.weight")
ckpt_out_proj_shape = _shape_of_state(state_dict, "out_proj.weight")
net_in_proj_shape = _shape_of_state(net_sd, "in_proj.weight")
net_out_proj_shape = _shape_of_state(net_sd, "out_proj.weight")

if ckpt_in_proj_shape != net_in_proj_shape:
    raise RuntimeError(
        f"[Cell8.4] ddpm_net input architecture mismatch: net={net_in_proj_shape}, ckpt={ckpt_in_proj_shape}"
    )
if ckpt_out_proj_shape != net_out_proj_shape:
    raise RuntimeError(
        f"[Cell8.4] ddpm_net output architecture mismatch: net={net_out_proj_shape}, ckpt={ckpt_out_proj_shape}"
    )

globals()["ddpm_net"].load_state_dict(state_dict, strict=True)

wrapper_device = getattr(active_wrapper, "device_obj", None)
if wrapper_device is not None:
    globals()["ddpm_net"].to(wrapper_device)

globals()["ddpm_net"].eval()
_rehydrate_ema_shadow(active_wrapper, ema_shadow_ckpt)

try:
    t_dim = int(globals()["ddpm_net"].t_embed.embedding_dim)
except Exception:
    t_dim = int(spec_json.get("t_emb_dim", ckpt_spec.get("t_emb_dim", 96)))

try:
    reg_dim = int(globals()["ddpm_net"].reg_embed.embedding_dim)
    reg_classes = int(globals()["ddpm_net"].reg_embed.num_embeddings)
except Exception:
    reg_dim = 16
    reg_classes = None

expected_in_width = int(ckpt_in_proj_shape[1])
tod_dim_inferred = int(expected_in_width - D_now - cond_dim_now - reg_dim - t_dim)

if tod_dim_inferred <= 0:
    raise RuntimeError(
        "[Cell8.4] Cannot infer positive TOD dimension from checkpoint: "
        f"in_proj={expected_in_width}, D={D_now}, cond_dim={cond_dim_now}, "
        f"reg_dim={reg_dim}, t_dim={t_dim}, inferred_tod_dim={tod_dim_inferred}"
    )

if int(D_now + cond_dim_now + reg_dim + tod_dim_inferred + t_dim) != expected_in_width:
    raise RuntimeError("[Cell8.4] Input-width decomposition failed.")

net_cond_dim = _infer_cond_dim_from_model(globals()["ddpm_net"])
if net_cond_dim is not None and int(net_cond_dim) != cond_dim_now:
    raise RuntimeError(
        f"[Cell8.4] active ddpm_net cond_dim mismatch: net={net_cond_dim}, active={cond_dim_now}"
    )

log(
    "[Cell8.4] OK: active DDPM runtime rehydrated from checkpoint | "
    f"in_proj={ckpt_in_proj_shape} | out_proj={ckpt_out_proj_shape} | tod_dim={tod_dim_inferred}"
)


# ----------------------------------------------------------
# 13) Advisory diagnostic + candidate registry validation
# ----------------------------------------------------------
candidate_registry = _normalize_registry_cols(candidate_registry, cols_now)

advisory_pass_cols = _extract_advisory_pass_cols(val_diag_audit, cols_now)
advisory_pass_cols = list(advisory_pass_cols)
advisory_excluded_cols = [c for c in cols_now if c not in set(advisory_pass_cols)]

_validate_col_subset("advisory_pass_cols", advisory_pass_cols, cols_now)
_check_unique_list("advisory_pass_cols", advisory_pass_cols)

best_candidate_by_col = _extract_best_candidate_by_col(val_diag_audit, cols_now)
decision_counts = _extract_decision_counts(val_diag_audit)
failure_reason_counts = _extract_failure_counts(val_diag_audit)

diag_D = val_diag_audit.get("D", D_now)
if diag_D is not None and int(diag_D) != D_now:
    raise RuntimeError(f"[Cell8.4] VAL diagnostic D mismatch: {diag_D} vs active={D_now}")

diag_role = str(val_diag_audit.get("ddpm_role", EXPECTED_DDPM_ROLE) or "").strip()
if diag_role and diag_role != EXPECTED_DDPM_ROLE:
    raise RuntimeError(f"[Cell8.4] VAL diagnostic role mismatch: {diag_role}")

diag_target_mode = str(val_diag_audit.get("target_mode", EXPECTED_TARGET_MODE) or "").strip()
if diag_target_mode and diag_target_mode != EXPECTED_TARGET_MODE:
    raise RuntimeError(f"[Cell8.4] VAL diagnostic target_mode mismatch: {diag_target_mode}")

if len(advisory_pass_cols) <= 0:
    msg = (
        "[Cell8.4] DDPM advisory diagnostic found zero pass columns. "
        "This is allowed: the downstream VAL-only selector must compare DDPM candidates against true A1 and may still reject all DDPM candidates."
    )
    if FAIL_IF_ADVISORY_PASS_ZERO:
        raise RuntimeError(msg)
    log("[Cell8.4][DDPM-ADVISORY][WARN] " + msg)

# Legacy safe-overwrite outputs are intentionally empty under v17.
# Cell 11 must use candidate registry + final VAL selector, not this legacy list.
DDPM_DOWNSTREAM_SAFE_OVERWRITE_COLS = []
DDPM_DOWNSTREAM_EXCLUDED_OVERWRITE_COLS = {
    str(c): {
        "reason": "legacy_direct_ddpm_overwrite_disabled_cell8_v17",
        "advisory_pass": bool(c in set(advisory_pass_cols)),
        "selection_authority": EXPECTED_SELECTION_AUTHORITY,
    }
    for c in cols_now
}

globals()["DDPM_CANDIDATE_REGISTRY"] = dict(candidate_registry)
globals()["DDPM_VAL_DIAGNOSTIC_AUDIT"] = dict(val_diag_audit)
globals()["DDPM_ADVISORY_PASS_COLS"] = list(advisory_pass_cols)
globals()["DDPM_ADVISORY_EXCLUDED_COLS"] = list(advisory_excluded_cols)
globals()["DDPM_DOWNSTREAM_SAFE_OVERWRITE_COLS"] = list(DDPM_DOWNSTREAM_SAFE_OVERWRITE_COLS)
globals()["DDPM_DOWNSTREAM_EXCLUDED_OVERWRITE_COLS"] = dict(DDPM_DOWNSTREAM_EXCLUDED_OVERWRITE_COLS)
globals()["DDPM_DOWNSTREAM_BEST_CANDIDATE_BY_COL"] = dict(best_candidate_by_col)

# Backward-compatible aliases only. Downstream cleaned cells should use
# DDPM_DOWNSTREAM_* names.
globals()["DDPM_CELL11_SAFE_OVERWRITE_COLS"] = list(DDPM_DOWNSTREAM_SAFE_OVERWRITE_COLS)
globals()["DDPM_CELL11_EXCLUDED_OVERWRITE_COLS"] = dict(DDPM_DOWNSTREAM_EXCLUDED_OVERWRITE_COLS)
globals()["DDPM_CELL11_BEST_CANDIDATE_BY_COL"] = dict(best_candidate_by_col)
globals()["DDPM_VAL_SCREEN_AUDIT"] = dict(val_diag_audit)  # backward-compatible alias only
globals()["DDPM_TARGET_SUPPORT_AUDIT"] = dict(target_support_audit)
globals()["DDPM_TARGET_TRANSFORM_AUDIT"] = dict(target_transform_audit)
globals()["DDPM_ROLE"] = EXPECTED_DDPM_ROLE
globals()["DDPM_TARGET_MODE"] = EXPECTED_TARGET_MODE
globals()["DDPM_SELECTION_AUTHORITY"] = EXPECTED_SELECTION_AUTHORITY
globals()["DDPM_LEGACY_DIRECT_OVERWRITE_DISABLED"] = True

log(
    "[Cell8.4] OK: DDPM advisory candidate contract validated | "
    f"advisory_pass={len(advisory_pass_cols)} | advisory_excluded={len(advisory_excluded_cols)} | "
    f"direct_safe_overwrite_cols=0 | candidate_families={candidate_registry['candidate_families_normalized']} | "
    f"decision_counts={decision_counts}"
)


# ----------------------------------------------------------
# 14) Runtime hyperparameters
# ----------------------------------------------------------
seq_len = int(spec_json.get("seq_len", ckpt_spec.get("seq_len", -1)))
stride = int(spec_json.get("stride", ckpt_spec.get("stride", -1)))
timesteps = int(spec_json.get("timesteps", ckpt_spec.get("timesteps", -1)))
epochs = int(spec_json.get("epochs", -1))
batch = int(spec_json.get("batch", -1))

if seq_len <= 0:
    raise RuntimeError(f"[Cell8.4] Invalid seq_len: {seq_len}")
if stride <= 0:
    raise RuntimeError(f"[Cell8.4] Invalid stride: {stride}")
if stride > seq_len:
    raise RuntimeError(f"[Cell8.4] Invalid stride/seq_len: stride={stride}, seq_len={seq_len}")
if timesteps < 10:
    raise RuntimeError(f"[Cell8.4] Invalid timesteps: {timesteps}")

CFG["ddpm_seq_len"] = int(seq_len)
CFG["ddpm_stride"] = int(stride)
CFG["ddpm_timesteps"] = int(timesteps)

net_width = int(spec_json.get("net_width", active_contract.get("net_width", -1)))
net_depth = int(spec_json.get("net_depth", active_contract.get("net_depth", -1)))
t_emb_dim_spec = int(spec_json.get("t_emb_dim", active_contract.get("t_emb_dim", t_dim)))
use_attention = bool(spec_json.get("use_attention", active_contract.get("use_attention", False)))
attention_heads = int(spec_json.get("attention_heads", active_contract.get("attention_heads", 0)))
dropout = float(spec_json.get("dropout", active_contract.get("dropout", 0.0)))

log(
    "[Cell8.4] OK: DDPM runtime hyperparameters locked | "
    f"seq_len={seq_len} | stride={stride} | timesteps={timesteps} | "
    f"width={net_width} | depth={net_depth} | attention={use_attention}"
)


# ----------------------------------------------------------
# 15) Device/runtime audit
# ----------------------------------------------------------
active_device = str(getattr(active_wrapper, "device", "unknown"))
active_device_obj = getattr(active_wrapper, "device_obj", None)

try:
    net_param_device = str(next(globals()["ddpm_net"].parameters()).device)
except Exception:
    net_param_device = "unknown"

active_device_canon = _canonical_torch_device_str(active_device)
active_device_obj_canon = _canonical_torch_device_str(active_device_obj)
net_param_device_canon = _canonical_torch_device_str(net_param_device)

cuda_available = bool(torch.cuda.is_available())
cuda_name = torch.cuda.get_device_name(0) if cuda_available else None

if active_device_canon.startswith("cuda") and not cuda_available:
    raise RuntimeError("[Cell8.4] Active wrapper claims CUDA but CUDA is unavailable.")

if active_device_obj is not None and net_param_device_canon != active_device_obj_canon:
    raise RuntimeError(
        "[Cell8.4] ddpm_net parameter device mismatch vs ddpm_seq.device_obj: "
        f"net={net_param_device} ({net_param_device_canon}) | "
        f"wrapper={active_device_obj} ({active_device_obj_canon})"
    )


# ----------------------------------------------------------
# 16) Legacy audit
# ----------------------------------------------------------
legacy_model_saved = globals().get("ddpm_model_legacy", None)
legacy_model_saved_type = type(legacy_model_saved).__name__
legacy_model_saved_cond_dim = _infer_cond_dim_from_model(legacy_model_saved)

legacy_wrapper_saved = globals().get("ddpm_legacy", None)
legacy_wrapper_saved_type = type(legacy_wrapper_saved).__name__
legacy_wrapper_saved_inner = getattr(legacy_wrapper_saved, "model", None)
legacy_wrapper_saved_inner_type = type(legacy_wrapper_saved_inner).__name__
legacy_wrapper_saved_inner_cond_dim = _infer_cond_dim_from_model(legacy_wrapper_saved_inner)

if legacy_model_saved_cond_dim is not None and int(legacy_model_saved_cond_dim) != cond_dim_now:
    msg = (
        "[Cell8.4] Preserved ddpm_model_legacy differs from active contract: "
        f"type={legacy_model_saved_type}, legacy_cond_dim={legacy_model_saved_cond_dim}, active={cond_dim_now}."
    )
    if FAIL_ON_LEGACY_DDPM_MISMATCH:
        raise RuntimeError(msg)
    log("[Cell8.4][LEGACY] " + msg)

if legacy_wrapper_saved_inner_cond_dim is not None and int(legacy_wrapper_saved_inner_cond_dim) != cond_dim_now:
    msg = (
        "[Cell8.4] Preserved ddpm_legacy inner model differs from active contract: "
        f"wrapper={legacy_wrapper_saved_type}, inner={legacy_wrapper_saved_inner_type}, "
        f"inner_cond_dim={legacy_wrapper_saved_inner_cond_dim}, active={cond_dim_now}."
    )
    if FAIL_ON_LEGACY_DDPM_MISMATCH:
        raise RuntimeError(msg)
    log("[Cell8.4][LEGACY] " + msg)


# ----------------------------------------------------------
# 17) Publish runtime audit
# ----------------------------------------------------------
DDPM_RUNTIME_AUDIT = {
    "version": "cell8_4_v15_THESIS_advisory_only_candidate_registry",
    "role": "sanity_check_and_downstream_gatekeeper",
    "ddpm_role": EXPECTED_DDPM_ROLE,
    "target_mode": EXPECTED_TARGET_MODE,
    "selection_authority": EXPECTED_SELECTION_AUTHORITY,
    "legacy_direct_overwrite_disabled": True,

    "sig12": sig12,
    "D": int(D_now),
    "ddpm_cols_sig_sha1": cols_sig_sha1_now,
    "ddpm_cols_sig_sha256": cols_sig_sha256_now,

    "ddpm_value_cols": list(cols_now),
    "ddpm_value_cols_subset_of_proto_contract": bool(
        "PROTO_COLS_CONTRACT" in globals()
        and set(cols_now).issubset(set(globals()["PROTO_COLS_CONTRACT"]))
    ),

    "cond_dim": int(cond_dim_now),
    "cond_sig": cond_sig_spec,
    "cond_spec_version": cond_ver_spec,

    "x_obs_basis": "Cell6_logical_value_finite",
    "x_obs_audit_path": xobs_audit_path,
    "x_obs_train_rate": float(xobs_audit["train"]["final_x_obs_rate"]),
    "x_obs_val_rate": float(xobs_audit["val"]["final_x_obs_rate"]),
    "x_obs_train_tier_obs_but_value_nan": int(xobs_audit["train"].get("cells_tier_obs_but_value_nan", -1)),
    "x_obs_val_tier_obs_but_value_nan": int(xobs_audit["val"].get("cells_tier_obs_but_value_nan", -1)),

    "val_diagnostic_audit_path": val_diag_path,
    "candidate_registry_path": candidate_registry_path,
    "target_support_audit_path": target_support_path,
    "target_transform_audit_path": target_transform_path,

    "advisory_pass_cols": list(advisory_pass_cols),
    "advisory_pass_cols_n": int(len(advisory_pass_cols)),
    "advisory_excluded_cols": list(advisory_excluded_cols),
    "advisory_excluded_cols_n": int(len(advisory_excluded_cols)),

    "val_diagnostic_decision_counts": decision_counts,
    "val_diagnostic_failure_reason_counts": failure_reason_counts,

    "candidate_families": list(candidate_registry["candidate_families_normalized"]),
    "candidate_registry_D": int(candidate_registry["D_normalized"]),

    "downstream_safe_overwrite_cols": [],
    "downstream_safe_overwrite_cols_n": 0,
    "downstream_excluded_overwrite_cols": dict(DDPM_DOWNSTREAM_EXCLUDED_OVERWRITE_COLS),
    "downstream_best_candidate_by_col": dict(best_candidate_by_col),
    
    # Deprecated aliases retained for compatibility only.
    "cell11_safe_overwrite_cols": [],
    "cell11_safe_overwrite_cols_n": 0,
    "cell11_excluded_overwrite_cols": dict(DDPM_DOWNSTREAM_EXCLUDED_OVERWRITE_COLS),
    "cell11_best_candidate_by_col": dict(best_candidate_by_col),

    "active_wrapper_global": "ddpm_seq",
    "active_model_global": "ddpm_net",
    "compat_wrapper_global": "ddpm",
    "compat_model_global": "ddpm_model",
    "active_model_family": "SeqDenoiser",

    "checkpoint_path": ckpt_path,
    "train_spec_path": spec_path,
    "scaler_path": scaler_path,
    "cond_spec_path": cond_spec_path,
    "active_contract_path": active_contract_path,
    "cell8_generator_contract_path": cell8_generator_contract_path,
    "cell8_generator_contract_validated": True,

    "in_proj_shape": list(ckpt_in_proj_shape),
    "out_proj_shape": list(ckpt_out_proj_shape),
    "input_width": int(expected_in_width),
    "tod_dim_inferred": int(tod_dim_inferred),
    "reg_dim": int(reg_dim),
    "reg_classes": int(reg_classes) if reg_classes is not None else None,
    "t_dim": int(t_dim),

    "seq_len": int(seq_len),
    "stride": int(stride),
    "timesteps": int(timesteps),
    "epochs": int(epochs),
    "batch": int(batch),

    "net_width": int(net_width),
    "net_depth": int(net_depth),
    "t_emb_dim": int(t_emb_dim_spec),
    "dropout": float(dropout),
    "use_attention": bool(use_attention),
    "attention_heads": int(attention_heads),

    "checkpoint_model_tensors": int(len(state_dict)),
    "checkpoint_ema_tensors": int(len(ema_shadow_ckpt)),
    "scaler_n_features_in": int(n_features),
    "scaler_type": type(ddpm_scaler_loaded).__name__,

    "COND_DIM_global": int(globals()["COND_DIM"]) if "COND_DIM" in globals() else None,

    "ddpm_alias_is_active_wrapper": bool(globals().get("ddpm", None) is active_wrapper),
    "ddpm_model_alias_is_active_net": bool(globals().get("ddpm_model", None) is globals()["ddpm_net"]),
    "ddpm_seq_net_is_ddpm_net": bool(getattr(active_wrapper, "net", None) is globals()["ddpm_net"]),

    "active_wrapper_device": active_device,
    "active_wrapper_device_canonical": active_device_canon,
    "active_wrapper_device_obj": str(active_device_obj),
    "active_wrapper_device_obj_canonical": active_device_obj_canon,
    "net_param_device": net_param_device,
    "net_param_device_canonical": net_param_device_canon,
    "cuda_available": bool(cuda_available),
    "cuda_name": cuda_name,

    "legacy_ddpm_model_type": legacy_model_saved_type,
    "legacy_ddpm_model_cond_dim": (
        int(legacy_model_saved_cond_dim) if legacy_model_saved_cond_dim is not None else None
    ),
    "legacy_ddpm_wrapper_type": legacy_wrapper_saved_type,
    "legacy_ddpm_inner_model_type": legacy_wrapper_saved_inner_type,
    "legacy_ddpm_inner_cond_dim": (
        int(legacy_wrapper_saved_inner_cond_dim) if legacy_wrapper_saved_inner_cond_dim is not None else None
    ),

    "rehydrated_model_state_from_checkpoint": True,
    "rehydrated_ema_shadow_from_checkpoint": True,

    "methodological_note": (
        "Cell 8.4 validates DDPM as an advisory-only conditional candidate generator. "
        "It intentionally exports no direct safe-overwrite columns. "
        "The downstream VAL-only selector is the only authority allowed to select "
        "DDPM candidates, and only if they beat true A1 under the frozen hybrid "
        "portfolio selection policy."
    ),
}

globals()["DDPM_RUNTIME_AUDIT"] = DDPM_RUNTIME_AUDIT
if "RUN_META" in globals():
    RUN_META.setdefault("runtime_audits", {})
    RUN_META["runtime_audits"]["DDPM_Cell8_4"] = DDPM_RUNTIME_AUDIT

runtime_audit_path = os.path.join(ARTDIR, "ddpm_runtime_audit_cell8_4.json")
_write_json(runtime_audit_path, DDPM_RUNTIME_AUDIT)

log(f"[Cell8.4] Saved runtime audit: {runtime_audit_path}")

log(
    "[Cell8.4] Active DDPM runtime validated: "
    f"sig12={sig12} | D={D_now} | cond_dim={cond_dim_now} | "
    f"model=SeqDenoiser | role={EXPECTED_DDPM_ROLE} | "
    f"selection_authority={EXPECTED_SELECTION_AUTHORITY} | "
    f"seq_len={seq_len} | stride={stride} | timesteps={timesteps} | "
    f"tod_dim={tod_dim_inferred} | input_width={expected_in_width} | "
    f"x_obs=Cell6_logical_value_finite | target_mode={EXPECTED_TARGET_MODE} | "
    f"device={active_device} | advisory_pass_cols={len(advisory_pass_cols)} | "
    f"direct_safe_overwrite_cols=0"
)

if advisory_pass_cols:
    log(f"[Cell8.4] DDPM advisory-pass columns: {advisory_pass_cols}")
else:
    log(
        "[Cell8.4] DDPM has zero advisory-pass columns. This is allowed. "
        "The downstream VAL-only selector must still evaluate DDPM candidate availability "
        "through the registry, but should select DDPM only if it beats true A1 on VAL."
    )

log("--- END:   Cell 8.4 — DDPM sanity checks (v15-THESIS advisory-only candidate registry) ---")