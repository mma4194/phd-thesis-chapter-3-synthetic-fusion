# %% CELL EXT.C.2 — Instrument agreement: pipeline per-target C2ST vs EXT per-column C2ST
# Question: do the two discriminators (pipeline QA vs EXT battery) RANK the 148
# continuous columns the same way, even though their absolute levels differ (0.68 vs 0.77)?
from scipy.stats import spearmanr, pearsonr

led = pd.read_csv(_pin("reports/cell12c6R_final_test_qa_metrics.csv"))
col_field = next(c for c in led.columns if c.lower() in ("col","column","target"))
auc_field = next(c for c in led.columns if "auc" in c.lower() or "c2st" in c.lower())
pipe = led.set_index(col_field)[auc_field].astype(float)

ext_rows = []
for c in COLSET:   # same 148 columns from EXT.C.1v3
    rr = pd.to_numeric(real_te[c], errors="coerce").dropna().to_numpy()
    ss = pd.to_numeric(syn_sci[c], errors="coerce").dropna().to_numpy()
    if len(rr) < 100 or len(ss) < 100: continue
    ext_rows.append((c, c2st_auc_blocked(rr, ss, n_blocks=4, max_rows=20_000)))
ext = pd.Series(dict(ext_rows), name="ext_auc")

joint = pd.concat([pipe, ext], axis=1, join="inner").dropna()
rho, p_rho = spearmanr(joint.iloc[:,0], joint.iloc[:,1])
r, p_r = pearsonr(joint.iloc[:,0], joint.iloc[:,1])
print(f"[EXT.C.2] n={len(joint)} | Spearman rho={rho:.3f} (p={p_rho:.1e}) | Pearson r={r:.3f}")
print(f"[EXT.C.2] level offset: pipeline mean={joint.iloc[:,0].mean():.4f} ext mean={joint.iloc[:,1].mean():.4f}")
joint.to_csv(EXT_OUT/"tables"/"EXT_C2_instrument_agreement.csv")
# Frozen reading: rho >= 0.8 -> instruments agree on ordering, absolute offset is calibration;
# 0.5-0.8 -> partial agreement, report both and do not substitute one for the other;
# < 0.5 -> instruments measure different things — investigate before citing either as "separability".