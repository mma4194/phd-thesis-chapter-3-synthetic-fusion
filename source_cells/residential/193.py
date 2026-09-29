# %% CELL EXT.B.1 — Gate battery v2 (KS + FIXED blocked C2ST; fail-loud, no silent NaN)
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

GATES = dict(ks_warn=0.20, ks_fatal=0.35, auc_warn=0.60, auc_fatal=0.70)

def c2st_auc_blocked(real, syn, n_blocks=8, max_rows=40_000):
    """Temporally blocked C2ST. FIX vs v1: real/syn rows from the SAME time block
    share the SAME group id, so every held-out fold contains both classes from
    unseen time. v1 gave real/syn different ids -> single-class folds -> silent NaN."""
    n = min(len(real), len(syn), max_rows)
    R, S = np.asarray(real, float)[:n], np.asarray(syn, float)[:n]
    if R.ndim == 1: R = R.reshape(-1, 1)
    if S.ndim == 1: S = S.reshape(-1, 1)
    X = np.vstack([R, S]); y = np.r_[np.zeros(n), np.ones(n)]
    reps = int(np.ceil(n / n_blocks))
    g = np.repeat(np.arange(n_blocks), reps)[:n]
    groups = np.r_[g, g]                                   # <- the fix
    X = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
    aucs = []
    for tr, te in GroupKFold(min(4, n_blocks)).split(X, y, groups):
        assert len(np.unique(y[te])) == 2, "[EXT.B.1] fold lost a class — grouping regression"
        m = LogisticRegression(max_iter=500).fit(X[tr], y[tr])
        a = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        aucs.append(max(a, 1 - a))
    assert aucs and np.isfinite(aucs).all(), "[EXT.B.1] non-finite C2ST folds — refusing silent NaN"
    return float(np.mean(aucs))

def gate_column(r, s):
    r = pd.to_numeric(r, errors="coerce").dropna()
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(r) < 100 or len(s) < 100:
        return dict(status="scope_excluded", ks=np.nan, auc=np.nan)
    ks = float(stats.ks_2samp(r, s).statistic)
    auc = c2st_auc_blocked(r.to_numpy(), s.to_numpy())
    status = "pass"
    if ks >= GATES["ks_fatal"] or auc >= GATES["auc_fatal"]:   status = "fatal"
    elif ks >= GATES["ks_warn"] or auc >= GATES["auc_warn"]:   status = "warning"
    return dict(status=status, ks=ks, auc=auc)

def audit_generator(name, scope_id, syn_df, scope_cols, real_df):
    rows = []
    for c in scope_cols:
        if c not in syn_df.columns or c not in real_df.columns:
            rows.append(dict(generator=name, scope=scope_id, column=c,
                             status="scope_excluded", ks=np.nan, auc=np.nan)); continue
        rows.append(dict(generator=name, scope=scope_id, column=c,
                         **gate_column(real_df[c], syn_df[c])))
    return pd.DataFrame(rows)

print("[EXT.B.1] gate battery v2 ready (KS + blocked C2ST, fail-loud).")