"""Validate frozen research inputs; do not reconstruct missing statistical formulas."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

KEY = ['anchor_name', 'signal']

def validate_manifest(root):
    root = Path(root)
    base = root / 'design_inputs'
    lock = json.loads((base / 'INPUT_LOCK.json').read_text())
    checks = []
    def check(name, ok, detail=''):
        checks.append({'check': name, 'status': 'PASS' if bool(ok) else 'FAIL', 'detail': str(detail)})
    for item in lock['files']:
        p = base / item['path']
        check('sha256:' + item['path'], p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'])
    if any(x['status'] != 'PASS' for x in checks):
        raise ValueError('Frozen manifest/supporting record identity changed: ' + str([x for x in checks if x['status'] != 'PASS']))
    frames = {}
    for name in ['all', 'tierA', 'tierB']:
        csv = pd.read_csv(base / f'coupling_manifest_{name}.csv')
        parquet = pd.read_parquet(base / f'coupling_manifest_{name}.parquet')
        parquet = parquet.where(parquet.notna(), np.nan)
        pd.testing.assert_frame_equal(csv, parquet, check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
        check('csv_parquet_agreement:' + name, True)
        check('unique_pairs:' + name, not csv.duplicated(KEY).any())
        frames[name] = csv.set_index(KEY).sort_index()
    allpairs, tier_a, tier_b = [frames[x] for x in ['all', 'tierA', 'tierB']]
    check('recorded_counts', (len(allpairs), len(tier_a), len(tier_b)) == (22, 5, 0))
    for name, tier in [('tierA', 'A'), ('tierB', 'B')]:
        pd.testing.assert_frame_equal(frames[name].reset_index(), allpairs[allpairs.manifest_tier.eq(tier)].reset_index(), check_dtype=False)
        check('subset:' + name, True)
    rep = pd.read_csv(base / 'supporting_records/alignment_replication_train_val.csv').set_index(KEY).sort_index()
    check('unique_replication_pairs', rep.index.is_unique)
    check('zigbee_pair_scope', set(allpairs.index) == set(rep[rep.tier.eq('zigbee')].index))
    shared = sorted(set(allpairs.columns) & set(rep.columns))
    pd.testing.assert_frame_equal(allpairs[shared], rep.loc[allpairs.index, shared], check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    check('manifest_to_replication_values', True, f'{len(shared)} shared fields across 22 pairs; absolute tolerance 1e-12')
    for split in ['train', 'val']:
        src = pd.read_csv(base / f'supporting_records/alignment_results_{split}.csv').set_index(KEY).sort_index()
        check('unique_alignment_pairs:' + split, src.index.is_unique)
        check('split_label:' + split, src['split'].eq(split).all())
        fields = [x for x in src.columns if x + '__' + split in rep]
        present = rep[rep['present_in_' + split].eq(1)]
        check('alignment_pair_scope:' + split, set(src.index) == set(present.index))
        values = present[[x + '__' + split for x in fields]].rename(columns={x + '__' + split: x for x in fields})
        pd.testing.assert_frame_equal(src[fields], values, check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
        check('alignment_to_replication_values:' + split, True, f'{len(fields)} fields; absolute tolerance 1e-12')
    meta = json.loads((base / 'coupling_manifest_meta.json').read_text())
    rules = meta['tierA_rules']; support = meta['support_rules']
    check('tierA_positive_flags', tier_a.manifest_ready_positive.eq(1).all() and tier_a.effect_direction.eq('positive').all())
    for split in ['train', 'val']:
        check('tierA_support:' + split, (tier_a['n_anchor_events__' + split] >= support['min_' + split + '_anchor_events']).all())
        check('tierA_labels:' + split, tier_a['alignment_label__' + split].isin(rules['required_labels']).all())
        check('tierA_positive_gain_sharpness:' + split, ((tier_a['delta_immediate_med__' + split] > 0) & (tier_a['sharpness_med__' + split] > 0)).all())
        check('tierA_lags:' + split, tier_a['peak_lag_sec__' + split].between(*rules['allowed_lag_range']).all())
        for col, threshold in [('z_immediate_vs_null', 'min_z_immediate_core'), ('z_sharpness_vs_null', 'min_z_sharpness_core'), ('coverage_modality_near', 'min_coverage_core')]:
            check('tierA_' + col + ':' + split, (tier_a[col + '__' + split] >= rules[threshold]).all())
    lagdiff = (tier_a.peak_lag_sec__train - tier_a.peak_lag_sec__val).abs()
    check('tierA_lag_agreement', (lagdiff <= rules['max_abs_lag_diff']).all() and np.allclose(lagdiff, tier_a.lag_agreement_abs_diff, rtol=0, atol=1e-12))
    check('tierA_lag_windows', ((tier_a.lag_lo <= tier_a.consensus_lag) & (tier_a.consensus_lag <= tier_a.lag_hi) & tier_a.lag_lo.between(*rules['allowed_lag_range']) & tier_a.lag_hi.between(*rules['allowed_lag_range'])).all())
    check('tierA_weights_finite_positive', (np.isfinite(tier_a.recommended_weight) & (tier_a.recommended_weight > 0)).all())
    result = {'status': 'PASS' if all(x['status'] == 'PASS' for x in checks) else 'FAIL', 'checks': checks,
              'input_lock_sha256': hashlib.sha256((base / 'INPUT_LOCK.json').read_bytes()).hexdigest(),
              'files': lock['files'], 'upstream_alignment_regenerated': False,
              'scope': 'Frozen-file identity, table consistency and recorded Tier A eligibility. Checks do not independently regenerate alignment statistics, weights or selection completeness.',
              'unverified_derivations': ['upstream alignment and null statistics', 'derived weights and replication scores', 'original builder implementation']}
    if result['status'] != 'PASS':
        raise ValueError('Manifest validation failed: ' + str([x for x in checks if x['status'] != 'PASS']))
    return result
