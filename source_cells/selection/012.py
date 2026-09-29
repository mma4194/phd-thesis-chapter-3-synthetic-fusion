# %% STRONG.9 — Smart* transfer claim scope: procedural transfer only

rows = []
if SMARTSTAR_ROOT is not None and SMARTSTAR_ROOT.exists():
    def smart_find(pat, required=False):
        hits = find_files(pat, [SMARTSTAR_ROOT])
        if not hits and required:
            raise FileNotFoundError(pat)
        return hits[0] if hits else None
    acct_path = smart_find("smartstar_feature_accounting_reconciled.csv")
    q4_path = smart_find("q4_scope_status.csv")
    q5_path = smart_find("q5_scope_status.csv")
    branch_path = smart_find("branch_readiness_summary.csv")
    rows.append({"component": "Smart* artifact root", "source": str(SMARTSTAR_ROOT.name), "allowed_transfer_claim": "procedural governance re-instantiation", "unsupported_claim": "numerical readiness transfer"})
    if acct_path:
        acct = pd.read_csv(acct_path)
        rows.append({"component": "role accounting", "source": str(acct_path.relative_to(SMARTSTAR_ROOT)), "allowed_transfer_claim": "role ownership and TRAIN/VAL/TEST eligibility accounting", "unsupported_claim": "CPS protocol-transfer evidence"})
    if q4_path:
        q4 = pd.read_csv(q4_path)
        rows.append({"component": "Q4", "source": str(q4_path.relative_to(SMARTSTAR_ROOT)), "allowed_transfer_claim": "unsupported-dimension declaration", "unsupported_claim": "physical-network coupling transfer"})
    else:
        rows.append({"component": "Q4", "source": "missing/not assessed", "allowed_transfer_claim": "no Q4 claim", "unsupported_claim": "physical-network coupling transfer"})
    if q5_path:
        rows.append({"component": "Q5", "source": str(q5_path.relative_to(SMARTSTAR_ROOT)), "allowed_transfer_claim": "unsupported/not assessed declaration", "unsupported_claim": "downstream utility transfer"})
    else:
        rows.append({"component": "Q5", "source": "missing/not assessed", "allowed_transfer_claim": "no Q5 claim", "unsupported_claim": "downstream utility transfer"})
else:
    rows.append({"component": "Smart*", "source": "SMARTSTAR_ARTIFACT_ROOT not set", "allowed_transfer_claim": "none from this notebook", "unsupported_claim": "transfer evidence"})

transfer = pd.DataFrame(rows)
write_csv(transfer, STRONG_TABLES / "smartstar_transfer_claim_scope.csv")
write_text(
    "# Smart* transfer claim scope\n\n"
    "Use Smart* as procedural governance-transfer evidence only. Because the selected Smart* traces contain no independent protocol stream, Q4 and Q5 do not test the CPS-specific coupling/utility components.\n",
    STRONG_DIR / "smartstar_transfer_claim_scope_note.md",
)
transfer
