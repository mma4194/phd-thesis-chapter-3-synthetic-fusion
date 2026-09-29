# %% CELL 11.x — STUDY-THESIS stale TEST-derived artifact preflight v2
# Purpose:
#   Fail closed on genuinely stale TEST-derived quarantine / repair / promotion artifacts.
#   Avoid false positives on explicit no-promotion governance files.
#
# Safe behavior:
#   - Does not delete, move, archive, or rewrite old artifacts.
#   - Writes a JSON report.
#   - Raises only for high-risk artifacts.
#
# Important:
#   Run this after Cell 11 and before Cell 12.a.
#   Do not run this while OUTDIR points to q6_public_reaudit_v1.

import os
import re
import json
from pathlib import Path
from datetime import datetime, timezone

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from IPython.display import display
except Exception:
    display = None


def _cell11x_log(msg: str) -> None:
    print(f"[Cell11.x stale-artifact-preflight-v2] {msg}")


def _cell11x_path(x):
    if not x:
        return None
    try:
        return Path(str(x)).expanduser().resolve()
    except Exception:
        return None


def _cell11x_cfg_get(*keys):
    cfg = globals().get("CFG", {})
    if not isinstance(cfg, dict):
        return None
    for key in keys:
        if cfg.get(key):
            return _cell11x_path(cfg.get(key))
    return None


outdir_11x = (
    _cell11x_path(globals().get("OUTDIR", None))
    or _cell11x_cfg_get("outdir", "OUTDIR", "output_dir")
    or _cell11x_path(os.environ.get("CPS_OUTDIR", ""))
)

out_syn_11x = (
    _cell11x_path(globals().get("OUT_SYN", None))
    or _cell11x_cfg_get("out_syn", "OUT_SYN", "synthetic_dir")
)

report_dir_11x = (
    _cell11x_path(globals().get("REPORT_DIR", None))
    or _cell11x_cfg_get("report_dir", "REPORT_DIR", "reports_dir")
)

if outdir_11x is None:
    raise RuntimeError(
        "[Cell11.x] OUTDIR is unresolved. Run Cell 1 and the root-reset cell first."
    )

if outdir_11x.name == "q6_public_reaudit_v1":
    raise RuntimeError(
        "[Cell11.x] OUTDIR still points to q6_public_reaudit_v1. "
        "Run the root-reset cell first. Do not use the Q6 public re-audit subdir as the main artifact root."
    )

if out_syn_11x is None:
    out_syn_11x = outdir_11x / "synthetic"

if report_dir_11x is None:
    report_dir_11x = outdir_11x / "reports"

scan_roots = []
for p in [outdir_11x, out_syn_11x, report_dir_11x]:
    if p is None:
        continue
    p = Path(p).expanduser().resolve()
    if p.exists() and str(p) not in {str(x) for x in scan_roots}:
        scan_roots.append(p)

if not scan_roots:
    raise RuntimeError("[Cell11.x] No existing artifact roots found to scan.")

report_dir_11x.mkdir(parents=True, exist_ok=True)

_cell11x_log("Scanning artifact roots:")
for root in scan_roots:
    _cell11x_log(f"  - {root}")


# Safe governance filenames that should not trigger failure.
SAFE_GOVERNANCE_PATTERNS = [
    re.compile(r"no[_-]?promotion[_-]?governance", re.I),
    re.compile(r"blocked[_-]?no[_-]?promotion", re.I),
    re.compile(r"report[_-]?only", re.I),
    re.compile(r"terminal[_-]?q4[_-]?report", re.I),
]


def _is_safe_governance_filename(name: str) -> bool:
    return any(p.search(name) for p in SAFE_GOVERNANCE_PATTERNS)


# Exact stale artifact names from old unsafe lineage.
STALE_EXACT_NAMES = {
    "a2_protocol_final_before_11b_quarantine.parquet",
    "a2_protocol_final_after_11b_quarantine.parquet",
    "a2_protocol_final_before_cell11b_quarantine.parquet",
    "a2_protocol_final_after_cell11b_quarantine.parquet",
    "a2_protocol_test_before_11b_quarantine.parquet",
    "a2_protocol_test_after_11b_quarantine.parquet",
    "a2_protocol_values_test_before_11b_quarantine.parquet",
    "a2_protocol_values_test_after_11b_quarantine.parquet",
    "cell11b_quarantine_manifest.json",
    "cell_11b_quarantine_manifest.json",
    "cell11b_test_quarantine_manifest.json",
    "cell_11b_test_quarantine_manifest.json",
}


HIGH_RISK_PATTERNS = [
    (
        re.compile(r"(cell[_-]?11b|11b).*(quarantine|quarantined|regression)", re.I),
        "old Cell 11b TEST-quarantine/regression artifact",
    ),
    (
        re.compile(r"(quarantine|quarantined).*(test|protocol|a2|cell[_-]?11b|11b)", re.I),
        "old TEST/protocol quarantine artifact",
    ),
    (
        re.compile(r"(before|pre)[_-]?(cell[_-]?)?11b.*(quarantine|quarantined)", re.I),
        "old pre-11b quarantine artifact",
    ),
    (
        re.compile(r"(after|post)[_-]?(cell[_-]?)?11b.*(quarantine|quarantined)", re.I),
        "old post-11b quarantine artifact",
    ),
    (
        re.compile(r"(testqa|test[_-]?qa).*(repair|revert|quarantine|quarantined|promotion|promoted)", re.I),
        "TEST-QA repair/revert/quarantine/promotion artifact",
    ),
    (
        re.compile(r"(repair|revert|quarantine|quarantined|promotion|promoted).*(testqa|test[_-]?qa)", re.I),
        "repair/revert/quarantine/promotion artifact tied to TEST-QA",
    ),
]


# Suspicious-but-possibly-safe filenames.
# These should be cleaned up before final submission, but they are not necessarily
# proof of leakage if the JSON contract says accepted=False and test_values_used_for_acceptance=False.
WARNING_PATTERNS = [
    (
        re.compile(r"(a0|zigbee|q4).*(accepted|promoted|promotion)", re.I),
        "Q4/A0 filename contains accepted/promoted/promotion language",
    ),
    (
        re.compile(r"(accepted|promoted|promotion).*(a0|zigbee|q4)", re.I),
        "filename contains accepted/promoted/promotion language tied to Q4/A0",
    ),
]


ARTIFACT_EXTENSIONS = {
    ".parquet",
    ".csv",
    ".json",
    ".pkl",
    ".pickle",
    ".npz",
    ".npy",
    ".feather",
    ".joblib",
    ".txt",
    ".md",
}


IGNORE_FILE_NAMES = {
    "study_thesis_stale_artifact_preflight_report.json",
    "study_thesis_stale_artifact_preflight_report_v2.json",
}


def _should_scan_file(path: Path) -> bool:
    if path.name.lower() in IGNORE_FILE_NAMES:
        return False
    if path.suffix.lower() not in ARTIFACT_EXTENSIONS:
        return False
    return True


def _read_json_if_possible(path: Path):
    if path.suffix.lower() != ".json":
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _json_contract_is_explicitly_not_accepted(obj) -> bool:
    if not isinstance(obj, dict):
        return False

    accepted = obj.get("accepted", None)
    accepted_outputs_written = obj.get("accepted_outputs_written", None)
    test_values_used = obj.get("test_values_used_for_acceptance", None)
    reason = str(obj.get("reason", "")).lower()

    return (
        accepted is False
        and accepted_outputs_written is False
        and test_values_used is False
        and (
            "cannot accept" in reason
            or "report" in reason
            or "test qa cannot" in reason
            or "no promotion" in reason
        )
    )


high_risk = []
warnings = []
scanned_n = 0

for root in scan_roots:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and d not in {"__pycache__", ".ipynb_checkpoints"}
            # Do not recursively scan nested Q6 public re-audit as part of main preflight.
            and d != "q6_public_reaudit_v1"
        ]

        for filename in filenames:
            path = Path(dirpath) / filename
            if not _should_scan_file(path):
                continue

            scanned_n += 1
            name_l = filename.lower()

            try:
                rel = str(path.relative_to(root))
            except Exception:
                rel = str(path)

            if _is_safe_governance_filename(filename):
                continue

            if name_l in STALE_EXACT_NAMES:
                high_risk.append({
                    "reason": "known stale artifact exact filename",
                    "relative_path": rel,
                    "path": str(path),
                })
                continue

            matched_high_risk = False
            for regex, reason in HIGH_RISK_PATTERNS:
                if regex.search(filename):
                    high_risk.append({
                        "reason": reason,
                        "relative_path": rel,
                        "path": str(path),
                    })
                    matched_high_risk = True
                    break

            if matched_high_risk:
                continue

            for regex, reason in WARNING_PATTERNS:
                if regex.search(filename):
                    obj = _read_json_if_possible(path)
                    if _json_contract_is_explicitly_not_accepted(obj):
                        warnings.append({
                            "reason": reason + " but JSON contract explicitly says accepted=False",
                            "relative_path": rel,
                            "path": str(path),
                            "action": (
                                "Rename this output in Cell 14.6 before final submission, "
                                "for example to terminal_q4_report_only_contract.json."
                            ),
                        })
                    else:
                        high_risk.append({
                            "reason": reason,
                            "relative_path": rel,
                            "path": str(path),
                        })
                    break


# Deduplicate.
high_risk = list({item["path"]: item for item in high_risk}.values())
warnings = list({item["path"]: item for item in warnings}.values())

report = {
    "cell": "CELL 11.x — STUDY-THESIS stale TEST-derived artifact preflight v2",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "status": "FAIL" if high_risk else "PASS_WITH_WARNINGS" if warnings else "PASS",
    "policy": {
        "delete_or_move_files": False,
        "fail_closed_on_high_risk_stale_artifacts": True,
        "warnings_allowed_for_local_verification": True,
        "test_data_used": False,
        "test_values_read": False,
        "safe_next_step_if_fail": "Use a clean CPS_OUTDIR or move high-risk files outside the submitted run directory.",
    },
    "scan_roots": [str(p) for p in scan_roots],
    "scanned_file_n": int(scanned_n),
    "high_risk_file_n": int(len(high_risk)),
    "warning_file_n": int(len(warnings)),
    "high_risk_files": high_risk,
    "warning_files": warnings,
}

report_path = report_dir_11x / "study_thesis_stale_artifact_preflight_report_v2.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

if warnings:
    _cell11x_log(f"Warnings: found {len(warnings)} filename/governance warning(s).")
    if pd is not None:
        df_w = pd.DataFrame(warnings)
        if display is not None:
            display(df_w[["reason", "relative_path", "action"]])
        else:
            print(df_w[["reason", "relative_path", "action"]].to_string(index=False))
    else:
        for item in warnings:
            print(f"WARNING: {item['reason']}: {item['path']}")

if high_risk:
    _cell11x_log(f"FAILED: found {len(high_risk)} high-risk stale artifact(s).")
    _cell11x_log(f"Report written to: {report_path}")

    if pd is not None:
        df_h = pd.DataFrame(high_risk)
        if display is not None:
            display(df_h[["reason", "relative_path", "path"]])
        else:
            print(df_h[["reason", "relative_path", "path"]].to_string(index=False))
    else:
        for item in high_risk:
            print(f"HIGH RISK: {item['reason']}: {item['path']}")

    raise RuntimeError(
        "[Cell11.x] High-risk stale TEST-derived artifacts detected. "
        "Stop here. Use a clean CPS_OUTDIR or move these files outside the submitted run directory."
    )

if warnings:
    _cell11x_log(
        f"PASS WITH WARNINGS: scanned {scanned_n} artifact file(s); "
        f"no high-risk stale artifacts found, but {len(warnings)} filename warning(s) remain."
    )
else:
    _cell11x_log(f"PASS: scanned {scanned_n} artifact file(s); no stale TEST-derived artifacts found.")

_cell11x_log(f"Report written to: {report_path}")

CFG["cell11x_stale_artifact_preflight_passed"] = True
CFG["cell11x_stale_artifact_preflight_report"] = str(report_path)

CELL11X_STALE_ARTIFACT_PREFLIGHT_REPORT = report
CELL11X_STALE_ARTIFACT_PREFLIGHT_PATH = str(report_path)