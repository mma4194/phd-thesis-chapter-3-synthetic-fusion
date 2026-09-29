# ==========================================================
# CELL 14.12 — Final Q4 no-promotion freshness / stale-evidence audit after protocol 11b
# v2.2 STUDY-THESIS-GRADE / NO-PROMOTION-FINAL-EVIDENCE-AWARE / GE1-BLOCKER
#
# Diagnostic only / no mutation
#
# Purpose:
# - Verify whether the FINAL authoritative Q4 no-promotion evidence was regenerated
#   after Cell 11b changed/quarantined the canonical A2 protocol artifact.
# - Detect stale/missing final Q4 governance reports/contracts by comparing
#   mtimes/hashes against the post-11b protocol anchor.
# - Treat final Q4 protocol/CPS coupled artifacts as NOT REQUIRED because
#   Cell 14.10/14.11 closed Q4 as blocked_no_promotion.
#
# Scientific contract:
# - Does not mutate artifacts.
# - Does not recompute Q4.
# - Does not read TEST values.
# - Does not promote/copy artifacts.
# - Only audits timestamps, file existence, hashes, and no-promotion consistency.
#
# Decision logic:
# - FAIL if final authoritative no-promotion Q4 files are missing or stale.
# - FAIL if any final governance contract indicates Q4 promotion/accepted artifact.
# - PASS_WITH_WARNING if final no-promotion files are fresh but optional
#   intermediate trace files are missing/stale.
# - PASS if authoritative no-promotion evidence is fresh and internally consistent.
# ==========================================================

import os
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime

log("--- START: Cell 14.12 — Final Q4 no-promotion freshness / stale-evidence audit after protocol 11b (v2.2) ---")

BASE = os.path.abspath(str(globals().get("OUTDIR", CFG.get("outdir", "."))))
SYN = os.path.join(BASE, "synthetic")
REP = os.path.join(BASE, "reports")
ART = os.path.join(BASE, "artifacts")
CONTRACT_DIR = os.path.join(ART, "contracts")

os.makedirs(REP, exist_ok=True)
os.makedirs(ART, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

CELL1412_VERSION = "cell14_12_q4_no_promotion_freshness_audit_v2_2_final_evidence_aware_ge1_blocker"

OUT_CSV = os.path.join(REP, "cell14_12_q4_no_promotion_freshness_audit.csv")
OUT_TRACE_CSV = os.path.join(REP, "cell14_12_optional_q4_traceability_audit.csv")
OUT_CONTENT_CSV = os.path.join(REP, "cell14_12_q4_no_promotion_content_contract_checks.csv")
OUT_JSON = os.path.join(ART, "cell14_12_q4_no_promotion_freshness_audit_summary.json")
OUT_CANONICAL_JSON = os.path.join(CONTRACT_DIR, "cell14_12_q4_no_promotion_freshness_audit_v2_2_THESIS.json")


# ----------------------------------------------------------
# 1) Helpers
# ----------------------------------------------------------
def exists(p):
    return bool(p and os.path.exists(str(p)))


def mtime(p):
    return float(os.path.getmtime(p)) if exists(p) else None


def mtimestr(p):
    if not exists(p):
        return None
    return datetime.fromtimestamp(os.path.getmtime(p)).isoformat(sep=" ", timespec="seconds")


def sha256_file(path, block=1 << 20):
    if not exists(path) or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_json(path):
    if not exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"__read_error__": f"{type(e).__name__}: {e}"}


def safe_csv(path):
    if not exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_float(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def safe_bool(x, default=False):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "t", "accepted", "pass", "promoted"}:
        return True
    if s in {"false", "0", "no", "n", "f", "", "nan", "none", "null", "rejected", "blocked"}:
        return False
    return default


def json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, set):
        return sorted([json_sanitize(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return json_sanitize(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return json_sanitize(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_sanitize(payload), f, indent=2, sort_keys=True)


def file_record(group, name, path, required_level, role):
    return {
        "group": group,
        "name": name,
        "path": path,
        "required_level": required_level,  # required_final | required_anchor | optional_trace | optional_candidate
        "role": role,
        "exists": exists(path),
        "mtime": mtime(path),
        "modified": mtimestr(path),
        "sha256": sha256_file(path) if str(path).endswith((".parquet", ".csv", ".json", ".txt")) else None,
    }


def newer_or_equal(t, anchor):
    if t is None or anchor is None:
        return None
    return bool(float(t) >= float(anchor))


def add_check(rows, name, passed, observed=None, expected=None, severity="required"):
    rows.append({
        "check": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "severity": severity,
    })


# ----------------------------------------------------------
# 2) Protocol freshness anchors
# ----------------------------------------------------------
protocol_inputs = {
    "cell11b_manifest": os.path.join(ART, "cell11b_protocol_test_regression_quarantine_manifest.json"),
    "cell11b_quarantined_A2": os.path.join(SYN, "A2_PROTOCOL_FINAL_QA_QUARANTINED.parquet"),
    "canonical_A2_PROTOCOL_FINAL": os.path.join(SYN, "A2_PROTOCOL_FINAL.parquet"),
    "canonical_A2_PROTOCOL_TEST": os.path.join(SYN, "A2_PROTOCOL_TEST.parquet"),
    "cell11_final_manifest": os.path.join(ART, "cell11_final_protocol_variant_manifest.json"),
    "cell11_quality_report": os.path.join(REP, "cell11_a0_a1_a2_protocol_quality_report.csv"),
}

anchor_rows = [
    file_record(
        group="protocol_anchor",
        name=name,
        path=path,
        required_level="required_anchor",
        role="post_11b_protocol_input_or_report",
    )
    for name, path in protocol_inputs.items()
]
anchor_df = pd.DataFrame(anchor_rows)
anchor_existing = anchor_df[anchor_df["exists"].astype(bool)].copy()
anchor_times = [x for x in anchor_existing["mtime"].tolist() if x is not None and pd.notna(x)]

freshness_anchor = max(anchor_times) if anchor_times else None
freshness_anchor_modified = (
    datetime.fromtimestamp(freshness_anchor).isoformat(sep=" ", timespec="seconds")
    if freshness_anchor is not None else None
)
anchor_source = None
if freshness_anchor is not None:
    tmp = anchor_existing[anchor_existing["mtime"].eq(freshness_anchor)]
    if len(tmp):
        anchor_source = str(tmp.iloc[0]["name"])


# ----------------------------------------------------------
# 3) Authoritative final Q4 no-promotion evidence files
# ----------------------------------------------------------
required_final_q4 = {
    "cell14_7_no_promotion_governance_csv": {
        "path": os.path.join(REP, "cell14_7_q4_no_promotion_governance.csv"),
        "role": "no_promotion_governance_ledger",
    },
    "cell14_7_no_promotion_contract": {
        "path": os.path.join(CONTRACT_DIR, "cell14_7_q4_no_promotion_governance_v2_1_THESIS.json"),
        "role": "canonical_no_promotion_governance_contract",
    },
    "cell14_8_A1_router_audit_contract": {
        "path": os.path.join(CONTRACT_DIR, "cell14_8_A1_router_audit_contract_v1_1_THESIS.json"),
        "role": "A1_router_audit_only_contract",
    },
    "cell14_9_A1a_deferral_contract": {
        "path": os.path.join(CONTRACT_DIR, "cell14_9_A1a_deferral_contract_v1_1_THESIS.json"),
        "role": "A1a_deferral_no_materialization_contract",
    },
    "cell14_10_final_q4_decision_ledger": {
        "path": os.path.join(REP, "cell14_10_final_q4_decision_ledger.csv"),
        "role": "final_q4_no_promotion_decision_ledger",
    },
    "cell14_10_final_q4_candidate_comparison": {
        "path": os.path.join(REP, "cell14_10_final_q4_candidate_comparison.csv"),
        "role": "final_q4_candidate_comparison_no_promotion",
    },
    "cell14_10_final_q4_publication_statement": {
        "path": os.path.join(REP, "cell14_10_final_q4_publication_statement.txt"),
        "role": "paper_ready_final_q4_no_promotion_statement",
    },
    "cell14_10_final_q4_decision_contract": {
        "path": os.path.join(CONTRACT_DIR, "cell14_10_final_q4_decision_contract_v1_1_THESIS.json"),
        "role": "canonical_final_q4_no_promotion_decision_contract",
    },
    "cell14_11_final_q4_evidence_table": {
        "path": os.path.join(REP, "cell14_11_final_q4_evidence_table.csv"),
        "role": "paper_ready_q4_evidence_table_no_promotion",
    },
    "cell14_11_scope_decision_table": {
        "path": os.path.join(REP, "cell14_11_final_q4_scope_decision_table.csv"),
        "role": "paper_ready_q4_scope_decision_table_no_promotion",
    },
    "cell14_11_paper_statement": {
        "path": os.path.join(REP, "cell14_11_final_q4_paper_statement.txt"),
        "role": "paper_ready_q4_scope_statement_no_promotion",
    },
    "cell14_11_contract": {
        "path": os.path.join(CONTRACT_DIR, "cell14_11_final_q4_scope_audit_contract_v1_1_THESIS.json"),
        "role": "canonical_final_q4_scope_audit_contract_no_promotion",
    },
}

# Optional candidates/intermediate traces.
optional_trace_q4 = {
    "cell14_6_A0_candidate_metrics": {
        "path": os.path.join(REP, "cell14_6_A0_candidate_manifest_q4_pair_metrics.csv"),
        "role": "optional_A0_candidate_metrics_not_final",
    },
    "cell14_6_A0_candidate_contract": {
        "path": os.path.join(CONTRACT_DIR, "cell14_6_A0_candidate_contract_v1_2_THESIS.json"),
        "role": "optional_A0_candidate_contract_not_final",
    },
    "cell14_6a_A0_fatal_blocker_diagnostic": {
        "path": os.path.join(REP, "cell14_6a_A0_fatal_blocker_diagnostic.csv"),
        "role": "optional_A0_fatal_blocker_diagnostic",
    },
    "cell14_6b_A0_policy_legality_contract": {
        "path": os.path.join(ART, "contracts", "cell14_6b_A0_policy_legality_contract_v1_0_THESIS.json"),
        "role": "optional_A0_policy_legality_contract",
    },
    "legacy_cell13_4_generic_q4_pair_metrics": {
        "path": os.path.join(REP, "cell13_4_generic_q4_pair_metrics.csv"),
        "role": "legacy_optional_generic_q4_trace",
    },
    "legacy_cell13_6_manifest_zigbee_pair_metrics": {
        "path": os.path.join(REP, "cell13_6_manifest_zigbee_pair_metrics.csv"),
        "role": "legacy_optional_manifest_zigbee_trace",
    },
}

rows = []
rows.extend(anchor_rows)

for name, spec in required_final_q4.items():
    rows.append(file_record(
        group="q4_final_authoritative_no_promotion",
        name=name,
        path=spec["path"],
        required_level="required_final",
        role=spec["role"],
    ))

for name, spec in optional_trace_q4.items():
    rows.append(file_record(
        group="q4_optional_trace",
        name=name,
        path=spec["path"],
        required_level="optional_trace",
        role=spec["role"],
    ))

audit = pd.DataFrame(rows)
audit["freshness_anchor_mtime"] = freshness_anchor
audit["freshness_anchor_modified"] = freshness_anchor_modified
audit["freshness_anchor_source"] = anchor_source
audit["is_newer_or_equal_to_anchor"] = audit["mtime"].apply(lambda x: newer_or_equal(x, freshness_anchor))
audit["is_older_than_anchor"] = audit["mtime"].apply(
    lambda x: bool(float(x) < float(freshness_anchor))
    if freshness_anchor is not None and x is not None and pd.notna(x)
    else None
)


# ----------------------------------------------------------
# 4) Content validation for no-promotion governance
# ----------------------------------------------------------
c147 = safe_json(required_final_q4["cell14_7_no_promotion_contract"]["path"])
c148 = safe_json(required_final_q4["cell14_8_A1_router_audit_contract"]["path"])
c149 = safe_json(required_final_q4["cell14_9_A1a_deferral_contract"]["path"])
c1410 = safe_json(required_final_q4["cell14_10_final_q4_decision_contract"]["path"])
c1411 = safe_json(required_final_q4["cell14_11_contract"]["path"])

evidence = safe_csv(required_final_q4["cell14_11_final_q4_evidence_table"]["path"])
scope = safe_csv(required_final_q4["cell14_11_scope_decision_table"]["path"])
ledger = safe_csv(required_final_q4["cell14_10_final_q4_decision_ledger"]["path"])

content_checks = []

# Cell 14.7 no-promotion.
add_check(
    content_checks,
    "cell14_7_contract_is_not_accepted",
    not safe_bool(c147.get("accepted", True), True),
    observed=c147.get("accepted"),
    expected=False,
)
add_check(
    content_checks,
    "cell14_7_contract_is_not_promoted",
    not safe_bool(c147.get("promoted", True), True),
    observed=c147.get("promoted"),
    expected=False,
)
add_check(
    content_checks,
    "cell14_7_candidate_outputs_not_promoted",
    not safe_bool(c147.get("strict_contract", {}).get("candidate_outputs_promoted_to_final", True), True),
    observed=c147.get("strict_contract", {}).get("candidate_outputs_promoted_to_final"),
    expected=False,
)
add_check(
    content_checks,
    "cell14_7_final_status_blocked_no_promotion",
    str(c147.get("summary", {}).get("governance_decision", {}).get("final_q4_status", "")) == "blocked_no_promotion",
    observed=c147.get("summary", {}).get("governance_decision", {}).get("final_q4_status"),
    expected="blocked_no_promotion",
)

# Cell 14.8 audit-only.
add_check(
    content_checks,
    "cell14_8_audit_only_no_direct_repair",
    not safe_bool(c148.get("expansion_authorization", {}).get("direct_repair_authorized_here", True), True),
    observed=c148.get("expansion_authorization", {}).get("direct_repair_authorized_here"),
    expected=False,
)

# Cell 14.9 A1a deferred.
add_check(
    content_checks,
    "cell14_9_A1a_deferred",
    safe_bool(c149.get("strict_contract", {}).get("A1a_deferred", False), False),
    observed=c149.get("strict_contract", {}).get("A1a_deferred"),
    expected=True,
)
add_check(
    content_checks,
    "cell14_9_no_materialization",
    not safe_bool(c149.get("strict_contract", {}).get("materialization_done_here", True), True),
    observed=c149.get("strict_contract", {}).get("materialization_done_here"),
    expected=False,
)

# Cell 14.10/14.11 final status.
add_check(
    content_checks,
    "cell14_10_final_status_blocked_no_promotion",
    str(c1410.get("final_decision", {}).get("final_q4_status", "")) == "blocked_no_promotion",
    observed=c1410.get("final_decision", {}).get("final_q4_status"),
    expected="blocked_no_promotion",
)
add_check(
    content_checks,
    "cell14_10_accepted_candidate_empty",
    str(c1410.get("final_decision", {}).get("accepted_candidate", "")) == "",
    observed=c1410.get("final_decision", {}).get("accepted_candidate"),
    expected="",
)
add_check(
    content_checks,
    "cell14_10_no_promotion",
    not safe_bool(c1410.get("strict_contract", {}).get("promotion_done_here", True), True),
    observed=c1410.get("strict_contract", {}).get("promotion_done_here"),
    expected=False,
)
add_check(
    content_checks,
    "cell14_11_final_status_blocked_no_promotion",
    str(c1411.get("final_q4_status", "")) == "blocked_no_promotion",
    observed=c1411.get("final_q4_status"),
    expected="blocked_no_promotion",
)
add_check(
    content_checks,
    "cell14_11_no_accepted_candidate",
    str(c1411.get("accepted_candidate", "")) == "",
    observed=c1411.get("accepted_candidate"),
    expected="",
)
add_check(
    content_checks,
    "cell14_11_A0_not_promoted",
    not safe_bool(c1411.get("A0_promoted", True), True),
    observed=c1411.get("A0_promoted"),
    expected=False,
)

# Evidence table should have no final candidate and A0 should be blocked.
if evidence.empty:
    add_check(content_checks, "evidence_table_not_empty", False, observed="empty", expected="nonempty")
else:
    final_candidate_n = int(evidence.get("final_publication_candidate", pd.Series([], dtype=bool)).fillna(False).astype(bool).sum())
    add_check(
        content_checks,
        "evidence_table_has_no_final_publication_candidate",
        final_candidate_n == 0,
        observed=final_candidate_n,
        expected=0,
    )

    if "candidate_scope" in evidence.columns:
        a0 = evidence[evidence["candidate_scope"].astype(str).eq("A0_zigbee_safe_candidate")]
        add_check(
            content_checks,
            "evidence_table_A0_candidate_row_present",
            len(a0) == 1,
            observed=len(a0),
            expected=1,
        )
        if len(a0):
            rec = a0.iloc[0].to_dict()
            add_check(
                content_checks,
                "evidence_table_A0_status_blocked_no_promotion",
                str(rec.get("status", "")) == "blocked_no_promotion",
                observed=rec.get("status"),
                expected="blocked_no_promotion",
            )
            # No-promotion requires at least one terminal publication blocker.
            # Earlier versions incorrectly required exactly two blockers, which
            # made the audit fail when the corrected final evidence had one
            # blocker. For the paper claim, the invariant is >=1 blocker:
            # any positive terminal blocker denies A0 promotion.
            a0_publication_blocker_n = safe_float(rec.get("publication_blocker_n"), default=np.nan)
            add_check(
                content_checks,
                "evidence_table_A0_publication_blockers_ge_one_for_no_promotion",
                bool(np.isfinite(a0_publication_blocker_n) and a0_publication_blocker_n >= 1.0),
                observed=rec.get("publication_blocker_n"),
                expected=">=1 under blocked_no_promotion",
            )
            if "final_publication_candidate" in rec:
                add_check(
                    content_checks,
                    "evidence_table_A0_final_publication_candidate_false",
                    not safe_bool(rec.get("final_publication_candidate"), default=True),
                    observed=rec.get("final_publication_candidate"),
                    expected=False,
                )

# Explicitly verify final Q4 coupled artifacts are absent/not required.
final_q4_protocol = os.path.join(SYN, "PROTOCOL_SYN_TEST_FINAL_Q4_COUPLED.parquet")
final_q4_cps = os.path.join(SYN, "CPS_SYNTHETIC_TEST_FINAL_Q4_COUPLED.parquet")
add_check(
    content_checks,
    "final_q4_protocol_artifact_not_required",
    True,
    observed={"exists": exists(final_q4_protocol), "path": final_q4_protocol},
    expected="not_required_under_blocked_no_promotion",
    severity="context",
)
add_check(
    content_checks,
    "final_q4_cps_artifact_not_required",
    True,
    observed={"exists": exists(final_q4_cps), "path": final_q4_cps},
    expected="not_required_under_blocked_no_promotion",
    severity="context",
)

content_check_df = pd.DataFrame(content_checks)


# ----------------------------------------------------------
# 5) Decision logic
# ----------------------------------------------------------
required_anchor = audit[audit["required_level"].eq("required_anchor")].copy()
required_final = audit[audit["required_level"].eq("required_final")].copy()
optional_trace = audit[audit["required_level"].eq("optional_trace")].copy()

missing_anchor = required_anchor[~required_anchor["exists"].astype(bool)].copy()
missing_final = required_final[~required_final["exists"].astype(bool)].copy()
stale_final = required_final[
    required_final["exists"].astype(bool)
    & required_final["is_older_than_anchor"].eq(True)
].copy()

cannot_determine_anchor = freshness_anchor is None

required_content_failures = content_check_df[
    content_check_df["severity"].eq("required") & ~content_check_df["passed"].astype(bool)
].copy() if not content_check_df.empty else pd.DataFrame()

missing_optional_trace = optional_trace[~optional_trace["exists"].astype(bool)].copy()
stale_optional_trace = optional_trace[
    optional_trace["exists"].astype(bool)
    & optional_trace["is_older_than_anchor"].eq(True)
].copy()

if cannot_determine_anchor:
    decision = "fail_cannot_determine_no_protocol_anchor"
    decision_severity = "fail"
elif len(missing_final) > 0:
    decision = "fail_missing_final_authoritative_no_promotion_q4_files"
    decision_severity = "fail"
elif len(stale_final) > 0:
    decision = "fail_stale_final_authoritative_no_promotion_q4_files"
    decision_severity = "fail"
elif len(required_content_failures) > 0:
    decision = "fail_final_no_promotion_q4_content_contract_failed"
    decision_severity = "fail"
elif len(missing_optional_trace) > 0 or len(stale_optional_trace) > 0:
    decision = "pass_final_q4_no_promotion_fresh_with_optional_traceability_warnings"
    decision_severity = "warning"
else:
    decision = "pass_final_q4_no_promotion_fresh"
    decision_severity = "pass"

final_q4_no_promotion_claim_current = bool(decision_severity in {"pass", "warning"})


# ----------------------------------------------------------
# 6) Save audit tables
# ----------------------------------------------------------
audit.to_csv(OUT_CSV, index=False)
optional_trace.to_csv(OUT_TRACE_CSV, index=False)
content_check_df.to_csv(OUT_CONTENT_CSV, index=False)


# ----------------------------------------------------------
# 7) Summary / contract
# ----------------------------------------------------------
evidence_brief = []
if not evidence.empty:
    keep_cols = [
        c for c in [
            "candidate_scope", "status", "final_publication_candidate",
            "protocol_scope", "pairs_total", "pairs_pass", "pairs_warning",
            "pairs_fatal", "publication_blocker_n", "mean_ETA_similarity",
            "mean_lag_peak_error", "mean_response_window_rate_error",
            "mean_manifest_profile_similarity", "decision",
        ]
        if c in evidence.columns
    ]
    evidence_brief = evidence[keep_cols].to_dict("records")

summary = {
    "cell": "14.12",
    "version": CELL1412_VERSION,
    "purpose": "freshness audit for final Q4 no-promotion evidence after Cell 11b protocol quarantine",
    "freshness_anchor_modified": freshness_anchor_modified,
    "freshness_anchor_source": anchor_source,
    "decision": decision,
    "decision_severity": decision_severity,
    "final_q4_no_promotion_claim_current": final_q4_no_promotion_claim_current,

    "required_final_files_n": int(len(required_final)),
    "missing_final_files_n": int(len(missing_final)),
    "stale_final_files_n": int(len(stale_final)),

    "optional_trace_files_n": int(len(optional_trace)),
    "missing_optional_trace_files_n": int(len(missing_optional_trace)),
    "stale_optional_trace_files_n": int(len(stale_optional_trace)),

    "missing_anchor_files": missing_anchor[["name", "path"]].to_dict("records"),
    "missing_final_files": missing_final[["name", "path"]].to_dict("records"),
    "stale_final_files": stale_final[["name", "path", "modified"]].to_dict("records"),
    "missing_optional_trace_files": missing_optional_trace[["name", "path"]].to_dict("records"),
    "stale_optional_trace_files": stale_optional_trace[["name", "path", "modified"]].to_dict("records"),

    "content_contract_checks": content_checks,
    "content_contract_required_failures": required_content_failures.to_dict("records"),

    "final_q4_status": "blocked_no_promotion",
    "accepted_candidate": "",
    "final_protocol_path": "",
    "final_cps_path": "",
    "final_q4_coupled_artifacts_required": False,

    "evidence_brief": evidence_brief,

    "reports": {
        "freshness_audit_csv": OUT_CSV,
        "optional_traceability_audit_csv": OUT_TRACE_CSV,
        "content_contract_checks_csv": OUT_CONTENT_CSV,
    },

    "synthetic_values_mutated": False,
    "q4_recomputed_here": False,
    "TEST_values_used": False,
    "artifact_promotion_done_here": False,

    "paper_interpretation": (
        "Final authoritative Q4 no-promotion evidence is fresh relative to the post-11b protocol artifact "
        "if decision_severity is pass or warning. Final Q4-coupled protocol/CPS artifacts are not required "
        "because final_q4_status is blocked_no_promotion."
    ),
}

write_json(OUT_JSON, summary)
write_json(OUT_CANONICAL_JSON, summary)


# ----------------------------------------------------------
# 8) Export globals
# ----------------------------------------------------------
globals()["CELL1412_VERSION"] = CELL1412_VERSION
globals()["CELL14_12_Q4_NO_PROMOTION_FRESHNESS_AUDIT_DF"] = audit
globals()["CELL14_12_Q4_NO_PROMOTION_CONTENT_CHECKS_DF"] = content_check_df
globals()["CELL14_12_Q4_NO_PROMOTION_FRESHNESS_SUMMARY"] = summary
globals()["CELL14_12_Q4_NO_PROMOTION_FRESHNESS_DECISION"] = decision
globals()["CELL14_12_Q4_NO_PROMOTION_FRESHNESS_DECISION_SEVERITY"] = decision_severity
globals()["CELL14_12_Q4_NO_PROMOTION_FRESHNESS_AUDIT_CSV"] = OUT_CSV
globals()["CELL14_12_Q4_NO_PROMOTION_CONTENT_CHECKS_CSV"] = OUT_CONTENT_CSV
globals()["CELL14_12_Q4_NO_PROMOTION_FRESHNESS_SUMMARY_JSON"] = OUT_JSON
globals()["CELL14_12_Q4_NO_PROMOTION_FRESHNESS_CANONICAL_JSON"] = OUT_CANONICAL_JSON


# ----------------------------------------------------------
# 9) Print/log
# ----------------------------------------------------------
print("\n=== CELL 14.12 Q4 NO-PROMOTION FRESHNESS AUDIT v2.2 ===")
print(audit.to_string(index=False, max_colwidth=150))

print("\n=== CELL 14.12 CONTENT CONTRACT CHECKS ===")
print(content_check_df.to_string(index=False, max_colwidth=150) if len(content_check_df) else "No content checks produced.")

print("\n=== CELL 14.12 SUMMARY ===")
print(json.dumps(json_sanitize(summary), indent=2))

print("\nSaved:")
print(OUT_CSV)
print(OUT_TRACE_CSV)
print(OUT_CONTENT_CSV)
print(OUT_JSON)
print(OUT_CANONICAL_JSON)

log(
    "[Cell14.12] Freshness decision: "
    f"{decision} | severity={decision_severity} | "
    f"missing_final={len(missing_final)} | stale_final={len(stale_final)} | "
    f"missing_optional_trace={len(missing_optional_trace)} | stale_optional_trace={len(stale_optional_trace)} | "
    f"final_q4_status=blocked_no_promotion"
)
log(
    "[Cell14.12] Contract flags | TEST_values_used=False | q4_recomputed_here=False | "
    "synthetic_values_mutated=False | artifact_promotion_done_here=False | "
    "final_q4_coupled_artifacts_required=False"
)
log("--- END: Cell 14.12 — Final Q4 no-promotion freshness / stale-evidence audit after protocol 11b (v2.2) ---")
