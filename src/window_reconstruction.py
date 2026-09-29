"""Explicit two-window reconstruction. Expected values enter comparison only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from workflow import ROOT,sha256,write_json

def compute(real,synthetic,scopes):
    if len(real)!=len(synthetic) or len(real)<2:raise ValueError('Unequal or empty inputs')
    edges=np.linspace(0,len(real),3).astype(int)
    rows=[];detail=[]
    for branch,cols in scopes.items():
        if not cols or len(cols)!=len(set(cols)):raise ValueError('Empty or duplicate feature scope')
        for w,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
            values=[]
            for col in cols:
                r=pd.to_numeric(real[col].iloc[a:b],errors='coerce').to_numpy(dtype=float)
                s=pd.to_numeric(synthetic[col].iloc[a:b],errors='coerce').to_numpy(dtype=float)
                r=r[np.isfinite(r)];s=s[np.isfinite(s)]
                if min(len(r),len(s))<100:raise ValueError(f'Insufficient finite observations: {branch}/{col}/{w}')
                ks=float(ks_2samp(r,s,alternative='two-sided',method='auto').statistic)
                values.append(ks);detail.append(dict(branch=branch,window=w,column=col,ks=ks,real_n=len(r),synthetic_n=len(s)))
            rows.append(dict(branch=branch,window=w,start_row=int(a),stop_row_exclusive=int(b),mean_ks=float(np.mean(values)),n_cols=len(cols)))
    return pd.DataFrame(rows),pd.DataFrame(detail)

def evaluate_windows(input_root,output):
    root=Path(input_root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    real_path=root/'synthetic/REAL_TEST_SPLIT.parquet'
    syn_path=root/'synthetic/CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet'
    registry_path=root/'reports/cell14_2_coupled_cps_column_registry.csv'
    driver_path=root/'reports/cell12e0_driver_target_contract.csv'
    reg=pd.read_csv(registry_path);drivers=pd.read_csv(driver_path)
    scopes={'router':sorted(reg.loc[reg.role.eq('protocol_router'),'col'].astype(str)),
            'zigbee':sorted(reg.loc[reg.role.eq('protocol_zigbee'),'col'].astype(str)),
            'drivers':sorted(drivers['col'].astype(str))}
    if {k:len(v) for k,v in scopes.items()}!={'router':18,'zigbee':4,'drivers':26}:raise ValueError('Scope differs from recorded 18/4/26 features')
    # Boundaries are an explicit reconstruction assumption, not inferred by fitting results.
    paths=[real_path,syn_path,registry_path,driver_path]
    policy={'implementation':'documented reconstruction; original cell unavailable','n_windows':2,
            'boundary_rule':'np.linspace(0,255540,3).astype(int); half-open row slices',
            'observation_policy':'finite values separately in each dataset; at least 100 per feature/window',
            'aggregation':'unweighted mean of per-feature two-sided KS statistics',
            'assumptions_not_historically_verified':['two equal contiguous windows','finite-value filtering and minimum count'],
            'scope':scopes,'input_sha256':{str(x):sha256(x) for x in paths}}
    write_json(out/'policy_before_computation.json',policy)
    cols=sorted(set(sum(scopes.values(),[])))
    real=pd.read_parquet(real_path,columns=cols+['sec_epoch_s__canon'])
    syn=pd.read_parquet(syn_path,columns=cols+['sec_epoch_s__canon'])
    if len(real)!=255540:raise ValueError('Expected TEST length 255540')
    rt=pd.to_numeric(real['sec_epoch_s__canon']).to_numpy();st=pd.to_numeric(syn['sec_epoch_s__canon']).to_numpy()
    if not np.array_equal(rt,st) or not np.all(np.diff(rt)==1):raise ValueError('Mismatched or discontinuous one-second grid')
    metrics,detail=compute(real,syn,scopes)
    metrics.to_csv(out/'window_metrics.csv',index=False);detail.to_csv(out/'per_feature_metrics.csv',index=False)
    expected=pd.read_csv(ROOT/'reference/current/window_metrics.csv')
    expected=expected[expected.branch.isin(scopes)]
    comparison=metrics.merge(expected,on=['branch','window'],suffixes=('_new','_reference'),validate='one_to_one')
    if len(comparison)!=6:raise ValueError('Expected six branch-window comparisons')
    comparison['absolute_difference']=(comparison.mean_ks_new-comparison.mean_ks_reference).abs()
    comparison['status']=np.where((comparison.absolute_difference<=1e-12)&(comparison.n_cols_new==comparison.n_cols_reference),'PASS','DIFFERENT')
    comparison.to_csv(out/'comparison.csv',index=False)
    result={'paper_agreement':'PASS_FOR_SIX_RECONSTRUCTED_WINDOW_ROWS' if comparison.status.eq('PASS').all() else 'DIFFERENT',
            'original_code_recovered':False,'input_root':str(root),'full_paper_verified':False}
    write_json(out/'result.json',result);return result
