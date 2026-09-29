# %% CELL EXT.A.1 — Build regime-scoped feature sets from disk artifacts (fail-closed, v2)
roles, role_p = load_role_manifest()
real_te, p_te = load_frame("real_test", EXPECTED_TEST_ROWS)
real_tr, p_tr = load_frame("real_train")
real_va, p_va = load_frame("real_val")
syn_sci, p_sci = load_frame("sci_artifact", EXPECTED_TEST_ROWS)
syn_pub, p_pub = load_frame("pub_artifact", EXPECTED_TEST_ROWS)
ts_col = guess_ts_col(real_te)
assert_timestamp_order(real_tr, real_va, real_te, ts_col)

# ---- INLINE ROLE-MAP DIAGNOSTIC (before we trust any role match) ----
roles = roles.map(lambda x: str(x).strip())          # coerce mixed int/str -> str, uniformly
from collections import Counter
role_hist = Counter(roles.values)
print(f"[EXT.A.1] role vocabulary ({len(role_hist)} distinct): {dict(role_hist)}")

# The IoT roles the regimes REQUIRE. If your manifest uses codes not names, these
# sets come back empty and we STOP — rather than silently building broken regimes.
def cols_of_role(roles, wanted):                      # local override: case-insensitive, str-safe
    want = {w.lower() for w in wanted}
    return [c for c, r in roles.items() if str(r).lower() in want]

cont_cols   = cols_of_role(roles, ["iot_continuous", "continuous"])
binary_cols = cols_of_role(roles, ["iot_binary", "binary"])
driver_all  = cols_of_role(roles, ["iot_driver", "sparse driver", "driver"])
excl_cols   = cols_of_role(roles, ["excluded", "placeholder"])

_role_report = {"continuous": len(cont_cols), "binary": len(binary_cols),
                "driver": len(driver_all), "excluded/placeholder": len(excl_cols)}
print(f"[EXT.A.1] role->column counts: {_role_report}")

# FAIL-CLOSED: if the role map did not populate the IoT branches, the manifest is
# using codes or a different vocabulary. Do NOT proceed on name-substring fallbacks.
if len(cont_cols) == 0 or len(driver_all) == 0:
    raise ValueError(
        "[EXT.A.1] Role map did not resolve IoT roles (continuous/driver empty).\n"
        f"  distinct role values seen: {sorted(role_hist)}\n"
        "  -> Your manifest likely stores role CODES, not names, OR the role field\n"
        "     name differs. Inspect one manifest entry and map codes->names in\n"
        "     load_role_manifest() before rerunning. Refusing to build regimes on\n"
        "     name-substring fallbacks (this is the Table-5 mismatched-scope trap).")

# Public logical columns (excluding persisted index)
pub_cols = [c for c in syn_pub.columns if not str(c).lower().startswith(("index", "__index", "unnamed"))]
print(f"[EXT.A.1] public logical columns on disk: {len(pub_cols)} (expected {EXPECTED_PUBLIC_LOGICAL_COLS})")
assert abs(len(pub_cols) - EXPECTED_PUBLIC_LOGICAL_COLS) <= 1, "Public baseline schema drift."

# Protocol columns: prefer role, fall back to name only as SUPPLEMENT (logged), never as sole source
protocol_role_cols = cols_of_role(roles, ["protocol"])
driver_cols = [c for c in driver_all if c in syn_pub.columns]
router_cols = sorted({c for c in syn_pub.columns
                      if "router" in str(c).lower()
                      or (c in protocol_role_cols and "zigbee" not in str(c).lower())})
zigbee_cols = sorted({c for c in syn_pub.columns if "zigbee" in str(c).lower()})
print(f"[EXT.A.1] protocol-by-role={len(protocol_role_cols)} | "
      f"router={len(router_cols)} zigbee={len(zigbee_cols)} driver(pub)={len(driver_cols)}")

REGIMES = {
    "R1_aggregate_gate": [c for c in syn_sci.columns if c not in set(excl_cols)],
    "R2_role_only":      sorted(set(driver_cols + router_cols + zigbee_cols + cont_cols + binary_cols)),
    "R3_governed":       sorted(set(driver_cols + router_cols + zigbee_cols)),
}
for k in REGIMES:
    REGIMES[k] = [c for c in REGIMES[k] if c != ts_col]
    print(f"[EXT.A.1] {k}: {len(REGIMES[k])} feature columns")

# Fail-closed on EACH regime's minimum viability, with role-specific messages
assert len(REGIMES["R3_governed"]) >= 20, \
    f"R3_governed too small ({len(REGIMES['R3_governed'])}); router/zigbee/driver mapping suspect."
assert len(cont_cols) >= 100, \
    f"Only {len(cont_cols)} continuous cols (paper reports 148). Role map incomplete — investigate."
assert len(REGIMES["R2_role_only"]) > len(REGIMES["R3_governed"]), \
    "R2 should strictly contain more than R3 (adds continuous+binary); it does not — role map broken."
print("[EXT.A.1] regime construction verified against role-map integrity checks.")