def profile_feature(name, s, epoch, row_start, cfg, metadata=None):
    """Describe values without assigning a final semantic owner."""
    metadata=metadata or {};n=len(s)
    null=s.isna().to_numpy(dtype=bool)
    numeric=is_numeric_dtype(s.dtype) or is_bool_dtype(s.dtype)
    valid=~null; nan=np.zeros(n,dtype=bool); inf=np.zeros(n,dtype=bool)
    if numeric:
        observed=s.iloc[np.flatnonzero(valid)].to_numpy()
        # Scalar integer/float/boolean only; decimal object arrays remain separately typed.
        try:
            obsfinite=np.isfinite(observed)
            obsnan=np.isnan(observed)
        except TypeError:
            numeric=False
        if numeric:
            ix=np.flatnonzero(valid)
            nan[ix]=obsnan; inf[ix]=~obsfinite & ~obsnan
            valid[ix]=obsfinite
    missing=null|nan
    nonmissing=~missing
    valid_idx=np.flatnonzero(valid)
    samples=[]
    def sample_rows(indices,kind,k=None):
        k=cfg['sample_count'] if k is None else k
        indices=np.asarray(indices,dtype=np.int64)
        if len(indices)>k:
            indices=stable_rng(name,kind,cfg['seed']).choice(indices,k,replace=False)
        for j in sorted(indices):
            samples.append({'feature_name':name,'sample_kind':kind,'source_row':int(row_start+j),
                            'epoch_seconds':int(epoch[j]),'timestamp_utc':pd.Timestamp(int(epoch[j]),unit='s',tz='UTC').isoformat(),
                            'value':show_value(s.iloc[j]),'stored_dtype':str(s.dtype)})
    sample_rows(np.arange(n),'random_all_rows')
    sample_rows(np.flatnonzero(nonmissing),'random_nonmissing')
    # Counts are over exact stored nonmissing values, not rounded representations.
    if numeric:
        vals=s.iloc[valid_idx].to_numpy()
        uniques,counts=np.unique(vals,return_counts=True)
        nunique=len(uniques)
        if len(vals)==0:rep='no_finite_numeric_values'
        elif nunique==1:rep='constant_numeric'
        elif np.isin(uniques,[0,1]).all():rep='binary_0_1_support'
        elif nunique==2:rep='two_point_numeric_not_0_1'
        elif np.all(vals==np.trunc(vals)):rep='integer_valued_numeric'
        else:rep='real_valued_numeric'
        if is_bool_dtype(s.dtype):rep='boolean_support' if nunique>1 else 'constant_boolean'
        zeros=int(np.sum(vals==0)); negative=int(np.sum(vals<0))
        positive_idx=valid_idx[vals>0];nonzero_idx=valid_idx[vals!=0]
        sample_rows(nonzero_idx,'random_nonzero')
        if len(vals):
            sample_rows([valid_idx[int(np.argmin(vals))]],'minimum',1)
            sample_rows([valid_idx[int(np.argmax(vals))]],'maximum',1)
        frequencies=[{'value':show_value(uniques[k]),'count':int(counts[k])} for k in np.argsort(-counts,kind='stable')[:cfg['top_values']]]
        all_support=[show_value(v) for v in uniques] if nunique<=cfg['support_limit'] else None
        pick=stable_rng(name,'distinct_values',cfg['seed']).choice(nunique,min(nunique,cfg['sample_count']),replace=False) if nunique else []
        distinct=[show_value(uniques[k]) for k in pick]
        # Adjacent finite pairs only: missing rows and non-1s gaps are never bridged.
        pair=np.flatnonzero(valid[:-1] & valid[1:] & (np.diff(epoch)==1))+1
        prev=s.iloc[pair-1].to_numpy();cur=s.iloc[pair].to_numpy()
        change=pair[cur!=prev]; rises=pair[cur>prev]; falls=pair[cur<prev]
        paired_zero=np.sum(cur==prev)
        # A decrease is a candidate reset, not proof of a reset or wraparound.
        pos=np.zeros(n,dtype=bool);pos[positive_idx]=True
        breaks=np.r_[True,np.diff(epoch)!=1]
        run_starts=np.flatnonzero(pos & (np.r_[True,~pos[:-1]]|breaks))
        run_ends=np.flatnonzero(pos & (np.r_[~pos[1:],True]|np.r_[np.diff(epoch)!=1,True]))
        lengths=run_ends-run_starts+1
        temporal={'finite_adjacent_pairs':len(pair),'changed_adjacent_pairs':len(change),'increases':len(rises),'decreases_candidate_resets':len(falls),
                  'unchanged_adjacent_pairs':int(paired_zero),'positive_run_count':len(lengths),
                  'median_positive_run_rows':float(np.median(lengths)) if len(lengths) else None,
                  'max_positive_run_rows':int(max(lengths)) if len(lengths) else None}
        if len(vals) and np.all(vals==np.trunc(vals)) and np.any(np.abs(vals.astype(np.longdouble))>2**53):
            precision_note='Integer magnitudes exceed 2^53: keep exact integer samples; float64 conversions may lose information.'
        else:precision_note='Samples preserve stored scalar values; upstream precision/encoding cannot be recovered here.'
        minv=show_value(vals.min()) if len(vals) else '';maxv=show_value(vals.max()) if len(vals) else ''
    else:
        # Preserve tuples, lists, strings and mixed representations rather than coercing.
        values=[s.iloc[j] for j in np.flatnonzero(nonmissing)]
        keys=[show_value(v) for v in values]
        freq=pd.Series(keys,dtype='string').value_counts()
        nunique=len(freq)
        nested=any(isinstance(v,(list,tuple,np.ndarray,dict)) for v in values)
        rep='nested_or_vector' if nested else ('datetime' if is_datetime64_any_dtype(s.dtype) else 'categorical_text_or_other')
        frequencies=[{'value':str(v),'count':int(c)} for v,c in freq.iloc[:cfg['top_values']].items()]
        support=list(freq.index.astype(str));all_support=support if nunique<=cfg['support_limit'] else None
        pick=stable_rng(name,'distinct_values',cfg['seed']).choice(nunique,min(nunique,cfg['sample_count']),replace=False) if nunique else []
        distinct=[support[k] for k in pick]
        positive_idx=[];change=[];rises=[];falls=[];zeros=negative=None;temporal={};minv=maxv=''
        precision_note='Raw representation retained; no automatic scalar encoding.'
    if missing.all():rep='all_missing'
    # Tiny exact differences around 0/1 remain visible; no 6-decimal rounding.
    windows=[]
    anchors={}
    for kind,indices in [('change',change),('decrease_candidate',falls),('positive',positive_idx),('missing',np.flatnonzero(missing))]:
        indices=np.asarray(indices,dtype=np.int64)
        if len(indices):
            anchors[kind]=int(stable_rng(name,'window_'+kind,cfg['seed']).choice(indices))
    for kind,anchor in anchors.items():
        for j in range(max(0,anchor-cfg['context_radius']),min(n,anchor+cfg['context_radius']+1)):
            windows.append({'feature_name':name,'window_kind':kind,'anchor_source_row':int(row_start+anchor),'source_row':int(row_start+j),
                            'timestamp_utc':pd.Timestamp(int(epoch[j]),unit='s',tz='UTC').isoformat(),'value':show_value(s.iloc[j])})
    flags=[]
    if missing.all():flags.append('NO_VALUES_IN_SELECTED_SPLIT_NOT_AUTOMATIC_EXCLUSION')
    if inf.any():flags.append('INFINITY_PRESENT')
    if semantic_hint(name) in {'global_event_or_reporting_aggregate_definition_needed','trigger_counter_definition_needed'}:flags.append('NEEDS_UPSTREAM_DEFINITION')
    if 'rgb' in name.lower():flags.append('CHECK_UPSTREAM_COLOUR_ENCODING')
    if rep in {'constant_numeric','constant_boolean'}:flags.append('CONSTANT_IN_SELECTED_SPLIT')
    if semantic_hint(name)=='wireless_metric_in_telemetry_namespace':flags.append('SEMANTIC_DOMAIN_AND_GENERATOR_ROLE_ARE_DIFFERENT_AXES')
    if metadata.get('pipeline_owner') in {'binary_state_target','iot_binary'} and numeric and len(valid_idx) and not np.isin(s.iloc[valid_idx].to_numpy(),[0,1]).all():flags.append('EXACT_SUPPORT_NOT_BINARY_0_1')
    summary={'feature_name':name,'blind_row_id':metadata.get('blind_row_id',''),'stored_dtype':str(s.dtype),'observed_representation':rep,
             'semantic_name_hint':semantic_hint(name),'existing_pipeline_owner':metadata.get('pipeline_owner',''),
             'existing_pipeline_reason':metadata.get('reason',''),'existing_target_cell':metadata.get('target_cell',''),
             'existing_continuous_family':metadata.get('family',''),'existing_synthesis_subfamily':metadata.get('synthesis_subfamily',''),
             'existing_routing_bucket':metadata.get('policy_bucket',''),'existing_stage12c_kind':metadata.get('expected_stage12c_kind',''),
             'rows_inspected':n,'null_or_nan_count':int(missing.sum()),'missing_fraction':float(missing.mean()),'infinity_count':int(inf.sum()),
             'finite_numeric_count':len(valid_idx) if numeric else None,'zero_count':zeros,'negative_count':negative,
             'unique_finite_values' if numeric else 'unique_nonmissing_values':nunique,
             'minimum_exact':minv,'maximum_exact':maxv,'flags':'|'.join(flags),**temporal}
    detail={**summary,'random_distinct_values':distinct,'complete_support_if_small':all_support,'most_frequent_values':frequencies,
            'precision_note':precision_note,'sample_population_note':'random_all_rows includes missingness; random_nonmissing is conditioned on a value being present; random_nonzero highlights rare nonzero values. None is an independent-event sample.'}
    return summary,detail,samples,windows


