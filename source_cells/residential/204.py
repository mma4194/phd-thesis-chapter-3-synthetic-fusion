# %% CELL EXT.OUT.1 — Manuscript-ready tables + reproducibility manifest for the EXT run
manifest = {
    "ext_version": "EXT.v1",
    "generated": pd.Timestamp.now().isoformat(),
    "rng_seed": RNG_SEED,
    "interpretation_rules_frozen_before_results": {
        "EXT.A": "governed regime gap must be smallest on permitted tasks; denied-scope TSTR degradation under R1/R2 is the measured governance value",
        "EXT.B": "per-generator ledgers reported side-by-side; no headline win counting outside compatible scopes",
        "EXT.C": "headline point estimates must lie within day-window envelope, else declare regime instability",
        "EXT.Q4H": "promotion of 6-pair subset iff zero blockers on untouched holdout under frozen policy",
    },
    "artifacts_consumed": {k: str(v[1]) for k, v in _ART_CACHE.items()},
    "outputs": sorted(str(p.relative_to(EXT_OUT)) for p in EXT_OUT.rglob("*") if p.is_file()),
}
with open(EXT_OUT / "manifests" / "EXT_run_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest["interpretation_rules_frozen_before_results"], indent=2))
print(f"[EXT.OUT] Manifest written. Send me: EXT_OUT/tables/*.csv, EXT_OUT/figures/*.png, "
      "and the full stdout of every EXT cell (including any fail-closed stops).")
