def focused_joint_review(cfg, output):
    """Aligned examples and empirical relations for pending/global features.
    Agreement of expressions is evidence of redundancy, not a semantic definition.
    """
    pf=pq.ParquetFile(Path(cfg['data_path']));names=set(pf.schema_arrow.names)
    focus=['iot__events_entity_unique','iot__events_total','iot__events_update_any','iot__any_update_raw','iot__tier_present','iot__obs_present',
           'iot__door_window_sensor_pid_005__sensor__trigger_count__value','iot__door_window_sensor_pid_007__sensor__trigger_count__value']
    focus += sorted(c for c in names if c.startswith(('iot__entity_obs__','iot__entity_stale__')) and any(t in c for t in ['door_window_sensor_pid_005','door_window_sensor_pid_007']))
    focus=list(dict.fromkeys(c for c in focus if c in names))
    if not focus:return pd.DataFrame()
    tcol=cfg['time_column'] or ('sec' if cfg['source_mode']=='canonical_fullgrid' else 'sec_epoch_s__canon')
    f=read_interval(pf,[tcol,*focus],0,cfg['n_train'],cfg['arrow_batch_rows'])
    anchors=set(stable_rng('joint','random_rows',cfg['seed']).choice(len(f),min(10,len(f)),replace=False).tolist())
    for col in focus:
        s=f[col];valid=s.notna().to_numpy(dtype=bool)
        pair=np.flatnonzero(valid[:-1]&valid[1:])+1
        cur=s.iloc[pair].to_numpy();prev=s.iloc[pair-1].to_numpy()
        for kind,hits in [('change',pair[cur!=prev]),('decrease',pair[cur<prev])]:
            if len(hits):
                chosen=stable_rng(col,kind+'_joint',cfg['seed']).choice(hits,min(2,len(hits)),replace=False)
                for j in chosen:anchors.update(range(max(0,int(j)-2),min(len(f),int(j)+3)))
    rows=[]
    for j in sorted(anchors):
        rows.append({'source_row':j,'timestamp_utc':pd.Timestamp(int(f[tcol].iloc[j]),unit='s',tz='UTC').isoformat(),
                     **{c:show_value(f[c].iloc[j]) for c in focus}})
    pd.DataFrame(rows).to_csv(output/'focused_aligned_examples.csv',index=False)
    checks=[]
    for x,y,kind in [('iot__events_update_any','iot__events_total','x == (y > 0)'),('iot__events_entity_unique','iot__events_total','x <= y'),('iot__events_update_any','iot__any_update_raw','x == y')]:
        if x not in f or y not in f:continue
        ix=np.flatnonzero(f[x].notna().to_numpy(dtype=bool)&f[y].notna().to_numpy(dtype=bool))
        a=f[x].iloc[ix].to_numpy();b=f[y].iloc[ix].to_numpy()
        finite=np.isfinite(a)&np.isfinite(b);a=a[finite];b=b[finite]
        ok=(a==(b>0)) if kind=='x == (y > 0)' else ((a<=b) if kind=='x <= y' else (a==b))
        checks.append({'x':x,'y':y,'check':kind,'paired_finite_rows':len(a),'matching_rows':int(ok.sum()),'match_fraction':float(ok.mean()) if len(a) else None,
                       'interpretation':'Observed TRAIN relationship only; not proof of construction, causality, or final role.'})
    out=pd.DataFrame(checks);out.to_csv(output/'focused_relationship_checks.csv',index=False)
    # The joint add-on is included in the same output manifest.
    mp=output/'inspection_manifest.json';meta=json.loads(mp.read_text())
    for p in [output/'focused_aligned_examples.csv',output/'focused_relationship_checks.csv']:
        meta['outputs'].append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    mp.write_text(json.dumps(meta,indent=2,default=str,allow_nan=False)+'\n')
    return out
