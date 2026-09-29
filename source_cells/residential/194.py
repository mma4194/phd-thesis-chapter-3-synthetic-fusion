# %% CELL EXT.B.2 — Cross-generator audit v2: ALL generators, per-scope, pipeline on same scopes
BASE = ARTIFACT_ROOT / "synthetic" / "baselines"
GEN_SCOPES = [   # (generator, scope_id, filename) — from the resolved candidate lists
    ("CTGAN",   "iot_continuous_compact",  "ctgan_iot_continuous_compact_tabular_synthetic_test.parquet"),
    ("TabDDPM", "iot_continuous_compact",  "tabddpm_iot_continuous_compact_tabular_synthetic_test.parquet"),
    ("TabDDPM", "public_candidate_core",   "tabddpm_public_candidate_core_tabular_synthetic_test.parquet"),
    ("TabDDPM", "protocol_core",           "tabddpm_protocol_core_tabular_synthetic_test.parquet"),
    ("TimeGAN", "protocol_sequence",       "timegan_protocol_sequence_temporal_synthetic_test.parquet"),
    ("TimeGAN", "sequence_core",           "timegan_sequence_core_temporal_synthetic_test.parquet"),
]

audits, scope_cols_map = [], {}
for gen, scope_id, fname in GEN_SCOPES:
    p = BASE / fname
    assert p.exists(), f"[EXT.B.2] pinned baseline missing: {p}"
    syn_g = pd.read_parquet(p)
    cols = [c for c in syn_g.columns
            if c in real_te.columns and not str(c).lower().startswith(("index", "__index", "unnamed"))]
    scope_cols_map.setdefault(scope_id, sorted(set(scope_cols_map.get(scope_id, [])) | set(cols)))
    print(f"[EXT.B.2] {gen:8s} {scope_id:24s} {len(cols)} cols <- {fname} sha256={sha256_file(p)[:12]}")
    audits.append(audit_generator(gen, scope_id, syn_g, cols, real_te))

# Role-aware pipeline audited on EACH scope's column set (same columns, same gates)
for scope_id, cols in scope_cols_map.items():
    audits.append(audit_generator("RoleAwarePipeline", scope_id, syn_sci, cols, real_te))

resB = pd.concat(audits, ignore_index=True)
assert resB.loc[resB.status.isin(["pass","warning","fatal"]), "auc"].notna().all(), \
    "[EXT.B.2] NaN AUC leaked into evaluable rows — C2ST regression"
ledger = resB.groupby(["scope","generator","status"]).size().unstack(fill_value=0)
means  = (resB[resB.status != "scope_excluded"]
          .groupby(["scope","generator"])[["ks","auc"]].mean().round(4))
resB.to_csv(EXT_OUT/"tables"/"EXT_B_crossgen_column_audit_v2.csv", index=False)
ledger.to_csv(EXT_OUT/"tables"/"EXT_B_crossgen_ledger_v2.csv")
means.to_csv(EXT_OUT/"tables"/"EXT_B_crossgen_means_v2.csv")
print(ledger.to_string()); print(); print(means.to_string())
print("[EXT.B.2] Frozen reading rule: per-scope ledgers reported side-by-side; "
      "no cross-scope headline counting.")