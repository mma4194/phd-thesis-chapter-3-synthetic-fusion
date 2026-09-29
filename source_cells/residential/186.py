# %% CELL EXT.DRV.GATE3 — legacy metric-degeneracy check retained as audit evidence
qa = CELL12E6_DRIVER_FINAL_TEST_QA_METRICS_DF.copy()
if {"event_rate_error", "burst_count_error", "overdispersion_error", "real_event_rate"}.issubset(qa.columns):
    tri = qa[["event_rate_error", "burst_count_error", "overdispersion_error"]].apply(pd.to_numeric, errors="coerce")
    print("[DRV.GATE3] max pairwise absolute differences among legacy triplet:")
    print((tri.sub(tri.iloc[:,0], axis=0).abs().max()).to_dict())
    idg = (pd.to_numeric(qa["real_overdispersion"], errors="coerce") - (1 - pd.to_numeric(qa["real_event_rate"], errors="coerce"))).abs().max() if "real_overdispersion" in qa.columns else np.nan
    print(f"[DRV.GATE3] max |real_overdispersion - (1-real_rate)| = {idg:.3e}")
print("[DRV.GATE3] Corrected ledger replaces the legacy triplet for release eligibility.")
