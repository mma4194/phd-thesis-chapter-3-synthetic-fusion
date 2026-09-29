"""Recompute documented results; saved measurements and new experiments stay distinct."""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import controlled
from toniot_metrics import robust_z_from_train, event_profile, profile_metrics

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'reference'

def compare_csv(actual,expected,keys,columns=None,rtol=1e-8,atol=1e-10):
    a=pd.read_csv(actual);b=pd.read_csv(expected)
    for x in (a,b):
        if x.duplicated(keys).any():raise ValueError(f'Duplicate comparison key: {Path(actual).name}')
    aa=a.set_index(keys).sort_index();bb=b.set_index(keys).sort_index()
    if not aa.index.equals(bb.index):raise ValueError(f'Row identities differ: {Path(actual).name}')
    cols=columns if columns is not None else list(bb.columns)
    missing=set(cols)-set(aa.columns)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    failures=[]
    for c in cols:
        x,y=aa[c],bb[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            equal=np.allclose(x.to_numpy(float),y.to_numpy(float),rtol=rtol,atol=atol,equal_nan=True)
        else:equal=x.fillna('<NA>').astype(str).equals(y.fillna('<NA>').astype(str))
        if not equal:failures.append(c)
    if failures:raise ValueError(f'Result mismatch in {Path(actual).name}: '+', '.join(failures))
    return {'rows':len(a),'compared_columns':len(cols),'rtol':rtol,'atol':atol}

def controlled_check(out):
    meta,_=controlled.run(out/'controlled')
    checks={}
    for fn,keys in [('trial_results.csv',['phase','features','candidate_c','trial','method']),('decision_by_setting.csv',['features','candidate_c','method']),('truth_by_decision.csv',['features','truth','method','phase'])]:
        # Trial and summary column names are read below; reject missing keys, never compare by row position.
        a=pd.read_csv(out/'controlled'/fn)
        if fn=='trial_results.csv':keys=['phase','features','candidate_c','trial','method']
        if fn=='truth_by_decision.csv':keys=[c for c in ['phase','features','truth','method'] if c in a.columns]
        checks[fn]=compare_csv(out/'controlled'/fn,REF/'controlled'/fn,keys)
    if meta['policy_tests']!=220 or meta['analytical_tests']!=16:raise ValueError('Unexpected test inventory')
    return {'scope':'Fresh controlled simulation and decision tests','checks':checks,'policy_tests':220,'analytical_checks':16,'trial_pairs':1600}

def smartstar_check(out):
    d=pd.read_csv(REF/'smartstar/test_evidence_q1_q2_q3_ledger.csv')
    rel=pd.read_csv(REF/'smartstar/q6_public_source_release_manifest.csv')
    policy=json.loads((REF/'smartstar/transfer_policy_manifest_v0.json').read_text())['readiness_gates']
    def threshold(v,p,w):return 'fatal' if not np.isfinite(v) else 'pass' if v<=p else 'warning' if v<=w else 'fatal'
    def ratio(v,p,w):return 'fatal' if not np.isfinite(v) else 'pass' if p[0]<=v<=p[1] else 'warning' if w[0]<=v<=w[1] else 'fatal'
    rows=[]
    for x in d.itertuples():
        role=x.owner_role.removeprefix('iot_');g=policy[role]
        if x.test_n_eval<10:grade='fatal'
        elif role=='continuous':grade=max([threshold(getattr(x,'test_'+m),g[k+'_pass_max'],g[k+'_warning_max']) for m,k in [('ks','ks'),('normalized_wasserstein','nw'),('acf1_absdiff','acf1_absdiff')]],key=['pass','warning','fatal'].index)
        elif role in ['binary','observability']:grade=max([threshold(getattr(x,'test_'+k),g[k+'_pass_max'],g[k+'_warning_max']) for k in ['rate_absdiff','transition_absdiff','run_ks']],key=['pass','warning','fatal'].index)
        else:grade=max([ratio(x.test_rate_ratio,g['rate_ratio_pass_range'],g['rate_ratio_warning_range']),ratio(x.test_burst_ratio,g['burst_ratio_pass_range'],g['burst_ratio_warning_range']),threshold(x.test_interarrival_ks,g['interarrival_ks_pass_max'],g['interarrival_ks_warning_max'])],key=['pass','warning','fatal'].index)
        rows.append({'feature':x.feature,'owner_role':x.owner_role,'test_status':grade})
    new=pd.DataFrame(rows);new.to_csv(out/'smartstar_grades.csv',index=False)
    compare_csv(out/'smartstar_grades.csv',REF/'smartstar/test_evidence_q1_q2_q3_ledger.csv',['feature'],['owner_role','test_status'])
    joined=new.merge(rel,on=['feature','owner_role'],suffixes=('_new','_stored'),validate='one_to_one')
    if len(joined)!=291 or not (joined.test_status_new==joined.test_status_stored).all():raise ValueError('Smart* selection join differs')
    if not np.array_equal(joined.release_decision.str.startswith('keep'),joined.test_status_new.isin(['pass','warning'])):raise ValueError('Smart* Q6 differs')
    counts=new.groupby(['owner_role','test_status']).size().unstack(fill_value=0)
    expected={'iot_continuous':[5,10,49],'iot_binary':[27,2,2],'iot_driver':[75,11,13],'iot_observability':[55,11,31]}
    for role,values in expected.items():
        if list(counts.loc[role,['pass','warning','fatal']])!=values:raise ValueError('Smart* paper counts differ')
    counts.to_csv(out/'smartstar_role_counts.csv')
    return {'scope':'Grades and selection recomputed from saved measurements, not raw traces','features':291,'pass':162,'warning':34,'fatal':95,'selected':196,'excluded':95}

def toniot_check(out, actual):
    ACTUAL=Path(actual)
    a=pd.read_parquet(ACTUAL/"intermediate/toniot_aligned_windows_with_split.parquet");d=pd.read_parquet(ACTUAL/"intermediate/toniot_physical_q4_driver_matrix.parquet")
    if len(a)!=7024 or len(d)!=len(a):raise ValueError('TON-IoT input shape differs')
    ref=a.split.isin(['TRAIN','VAL']);test=a.split.eq('TEST')
    frozen=pd.read_csv(ACTUAL/"physical_q4/manifests/toniot_physical_q4_manifest_trainval_frozen.csv")
    ts=pd.to_datetime(a.window_start,utc=True);rows=[];clock=[]
    if not ts.is_unique or not ts.is_monotonic_increasing:raise ValueError('Invalid window order')
    def clock_profile(ev,y,t):
        edge=np.flatnonzero(ev>0);edge=edge[(edge>=5)&(edge+5<len(y))]
        tt=np.asarray(t.astype('int64'));use=[i for i in edge if np.all(np.diff(tt[i-5:i+6])==60_000_000_000)]
        return (np.mean(np.stack([y[i-5:i+6] for i in use]),axis=0) if use else None),len(use),len(edge)-len(use)
    for x in frozen.itertuples():
        er=d.loc[ref,x.driver_column].fillna(0).to_numpy(int);et=d.loc[test,x.driver_column].fillna(0).to_numpy(int)
        yr=a.loc[ref,x.protocol_response_column].fillna(0);yt=a.loc[test,x.protocol_response_column].fillna(0)
        zr=robust_z_from_train(yr,yr);zt=robust_z_from_train(yr,yt)
        pr,nr=event_profile(er,zr);pt,nt=event_profile(et,zt)
        m=profile_metrics(pr,pt) if nt>=5 else dict(eta_similarity=np.nan,lag_error_windows=np.nan,response_window_rel_error=np.nan,status='blocker')
        rows.append(dict(pair_id=x.pair_id,reference_events_trainval=nr,test_events=nt,**{'test_'+k:v for k,v in m.items()}))
        cr,ncr,gr=clock_profile(er,zr,ts[ref]);ct,nct,gt=clock_profile(et,zt,ts[test])
        mm=profile_metrics(cr,ct) if ncr>=5 and nct>=5 else dict(eta_similarity=np.nan,lag_error_windows=np.nan,response_window_rel_error=np.nan,status='insufficient')
        zero=bool(cr is not None and ct is not None and np.allclose(cr,0,atol=1e-12,rtol=0) and np.allclose(ct,0,atol=1e-12,rtol=0))
        clock.append(dict(pair_id=x.pair_id,reference_events=ncr,test_events=nct,reference_gap_windows_removed=gr,test_gap_windows_removed=gt,both_profiles_zero=zero,**mm))
    new=pd.DataFrame(rows);new.to_csv(out/'toniot_pair_metrics.csv',index=False)
    compare_csv(out/'toniot_pair_metrics.csv',REF/'toniot/toniot_physical_q4_test_event_response_audit.csv',['pair_id'],list(new.columns.drop('pair_id')),rtol=1e-7,atol=1e-9)
    counts=new.test_status.value_counts().to_dict()
    if counts!={'blocker':44,'warning':6}:raise ValueError('TON-IoT paper grades differ')
    pd.DataFrame(clock).to_csv(out/'toniot_clock_sensitivity.csv',index=False)
    q6=pd.read_csv(ACTUAL/"physical_q4/tables/toniot_q6_release_scope_table_physical_q4.csv");keep=q6.q6_keep.astype(str).str.lower().eq('true')
    if int(keep.sum())!=185 or len(q6)!=186:raise ValueError('TON-IoT Q6 inventory differs')
    # Counts verify the Q6 manifest produced by the current raw-source stage.
    return {'scope':'Q4 and clock measurements from current aligned data and pairs; current Q6 manifest accounting','pairs':50,'pass':0,'warning':6,'blocker':44,'selected':185,'excluded':1,'clock_sensitivity':pd.DataFrame(clock).status.value_counts().to_dict(),'test_support_below_five':int((new.test_events<5).sum())}

