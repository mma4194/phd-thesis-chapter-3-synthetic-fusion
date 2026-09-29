def inspect_parquet(cfg, embedded_reference):
    path=Path(cfg['data_path']).expanduser().resolve()
    if not path.is_file():raise FileNotFoundError(f'Parquet not found: {path}. Edit DATA_PATH in configuration. Do not rerun the synthesis notebook.')
    pf=pq.ParquetFile(path);names=pf.schema_arrow.names
    if len(names)!=len(set(names)):raise ValueError('Duplicate Parquet feature names.')
    if cfg['source_mode'] not in {'canonical_fullgrid','persisted_train_split'}:raise ValueError('Choose a documented source_mode.')
    if cfg['strict_source_contract']:
        expected_rows=cfg['expected_total_rows'] if cfg['source_mode']=='canonical_fullgrid' else cfg['n_train']
        if pf.metadata.num_rows!=expected_rows:raise ValueError(f'Row count {pf.metadata.num_rows} != expected {expected_rows}; verify source identity.')
        if cfg['source_mode']=='canonical_fullgrid' and len(names)!=cfg['expected_columns']:raise ValueError('Column count differs from original canonical source (740).')
    start=0;stop=cfg['n_train']
    if stop>pf.metadata.num_rows:raise ValueError('TRAIN length exceeds source rows.')
    tcol=cfg['time_column'] or ('sec' if cfg['source_mode']=='canonical_fullgrid' else 'sec_epoch_s__canon')
    if tcol not in names:raise ValueError(f'Time column {tcol} is absent; set TIME_COLUMN explicitly.')
    t=read_interval(pf,[tcol],start,stop,cfg['arrow_batch_rows'])[tcol]
    if t.isna().any():raise ValueError('Missing timestamps.')
    arr=t.to_numpy()
    if not np.isfinite(arr).all() or not np.equal(arr,np.trunc(arr)).all():raise ValueError('Timestamp must be integer Unix seconds.')
    epoch=arr.astype(np.int64)
    if not (np.diff(epoch)==1).all():raise ValueError('TRAIN is not ordered gap-free 1 Hz; preserve order and investigate rather than sorting silently.')
    if cfg['strict_source_contract'] and int(epoch[0])!=cfg['expected_first_epoch']:raise ValueError('Source first timestamp differs from saved original contract.')
    roles,role_sources=load_existing_roles(cfg['run_root'],embedded_reference)
    if cfg['scope']=='all':selected=list(names)
    elif cfg['scope']=='iot':selected=[c for c in names if c.startswith(('iot__','events_in_sec__','telemetry_in_sec__'))]
    elif cfg['scope']=='m7':selected=[r['feature_name'] for r in embedded_reference]
    elif cfg['scope']=='custom':selected=list(cfg['custom_features'])
    else:raise ValueError('scope must be all, iot, m7 or custom.')
    if len(selected)!=len(set(selected)) or not selected:raise ValueError('Feature selection is empty or duplicated.')
    absent=[c for c in selected if c not in names]
    if absent:raise ValueError(f'Selected features absent from source: {absent[:20]}')
    # Use a fresh inspection directory so prior reviews and pipeline artefacts cannot be overwritten.
    parent=Path(cfg['output_parent']).expanduser().resolve()
    output=parent/('feature_review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    meta={'source':str(path),'source_mode':cfg['source_mode'],'source_bytes':path.stat().st_size,'source_mtime_ns':path.stat().st_mtime_ns,
          'source_schema_sha256':hashlib.sha256(str(pf.schema_arrow).encode()).hexdigest(),'source_file_sha256':None,
          'expected_original_sha256':cfg.get('expected_input_sha256'),'source_hash_verified':False,
          'split':'TRAIN','source_row_start_inclusive':start,'source_row_stop_exclusive':stop,'first_epoch':int(epoch[0]),'last_epoch':int(epoch[-1]),
          'source_columns':len(names),'selected_features':len(selected),'seed':cfg['seed'],'sample_count':cfg['sample_count'],'existing_role_sources':role_sources,
          'role_changes_applied':False,'test_values_used_in_profiles':False,'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in cfg.items()},
          'status':'RUNNING'}
    if cfg.get('compute_source_sha256'):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        meta['source_file_sha256']=h.hexdigest()
        if cfg['source_mode']=='canonical_fullgrid' and cfg.get('expected_input_sha256'):
            if h.hexdigest()!=cfg['expected_input_sha256']:raise ValueError('Full source hash differs from original; do not interpret as same dataset.')
            meta['source_hash_verified']=True
    (output/'inspection_manifest.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
    summaries=[];details=[];samples=[];windows=[]
    print(f'Inspecting {len(selected)} fields, TRAIN rows {start:,}–{stop-1:,}; reading {cfg["column_group_size"]} columns at a time.')
    for offset in range(0,len(selected),cfg['column_group_size']):
        group=selected[offset:offset+cfg['column_group_size']]
        frame=read_interval(pf,group,start,stop,cfg['arrow_batch_rows'])
        for name in group:
            summary,detail,ss,ww=profile_feature(name,frame[name],epoch,start,cfg,roles.get(name,{}))
            summaries.append(summary);details.append(detail);samples.extend(ss);windows.extend(ww)
            if cfg['print_every_feature']:
                randoms=[r['value'] for r in ss if r['sample_kind']=='random_nonmissing']
                print(f'\n[{len(summaries)}/{len(selected)}] {name} | {summary["blind_row_id"]} | {summary["observed_representation"]} | missing={summary["null_or_nan_count"]}/{len(frame)}')
                print('  Existing owner:',summary['existing_pipeline_owner'] or '(not supplied)','| Name hint:',summary['semantic_name_hint'])
                print('  Random nonmissing values:',randoms or '(none; see missingness and random_all_rows)')
                print('  Random distinct values:',detail['random_distinct_values'])
        del frame;gc.collect()
        pd.DataFrame(summaries).to_csv(output/'feature_inventory_PARTIAL.csv',index=False)
        print(f'Progress: {len(summaries)}/{len(selected)} features',flush=True)
    inventory=pd.DataFrame(summaries)
    inventory.to_csv(output/'feature_inventory.csv',index=False)
    pd.DataFrame(samples).to_csv(output/'feature_samples.csv',index=False)
    pd.DataFrame(windows).to_csv(output/'feature_temporal_examples.csv',index=False)
    (output/'feature_details.json').write_text(json.dumps(details,indent=2,default=str,allow_nan=False)+'\n')
    inventory.groupby(['observed_representation','semantic_name_hint'],dropna=False).size().rename('features').reset_index().to_csv(output/'category_counts.csv',index=False)
    manual=inventory[['blind_row_id','feature_name','stored_dtype','observed_representation','semantic_name_hint','existing_pipeline_owner','flags']].copy()
    for c in ['reviewed_semantic_role','reviewed_representation','reviewed_eligibility','review_rationale','source_definition_reference']:
        manual[c]=''
    manual.to_csv(output/'manual_feature_review.csv',index=False)
    write_html(output/'feature_review.html',summaries,details,samples,windows,'TRAIN only · '+str(len(selected))+' features · seed '+str(cfg['seed']))
    # Plain-text companion is easy to print without notebook output truncation.
    with (output/'feature_review.txt').open('w',encoding='utf-8') as f:
        for detail in details:
            f.write(json.dumps(detail,ensure_ascii=False,indent=2,default=str)+'\n\n')
            for sample in samples:
                if sample['feature_name']==detail['feature_name'] and sample['sample_kind']=='random_nonmissing':
                    f.write(f"{sample['source_row']} | {sample['timestamp_utc']} | {sample['value']}\n")
    meta['status']='COMPLETE_INSPECTION_NOT_ADJUDICATION';meta['completed_features']=len(inventory)
    meta['outputs']=[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.iterdir()) if p.is_file() and p.name not in {'inspection_manifest.json','feature_inventory_PARTIAL.csv'}]
    (output/'inspection_manifest.json').write_text(json.dumps(meta,indent=2,default=str,allow_nan=False)+'\n')
    (output/'feature_inventory_PARTIAL.csv').unlink(missing_ok=True)
    print('\nComplete. Open:',output/'feature_review.html')
    print('Return feature_inventory.csv, feature_samples.csv, feature_temporal_examples.csv and inspection_manifest.json for review.')
    return output,inventory


