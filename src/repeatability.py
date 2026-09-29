"""Compare scientific outputs from two fresh runs, independently of references."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

from workflow import write_json


def parquet_digest(path):
    import pyarrow.parquet as pq
    table = pq.ParquetFile(path)
    digest = hashlib.sha256()
    digest.update(json.dumps([(x.name, str(x.type)) for x in table.schema_arrow]).encode())
    # Fixed batch boundaries; ignore compression and Parquet metadata timestamps.
    for batch in table.iter_batches(batch_size=32768):
        frame = batch.to_pandas()
        digest.update(pd.util.hash_pandas_object(frame, index=False, categorize=False)
                      .to_numpy(dtype='<u8').tobytes())
    return {'sha256_ordered_values': digest.hexdigest(),
            'rows': table.metadata.num_rows, 'columns': table.schema_arrow.names}


def baseline_training_status(run):
    path = Path(run) / 'residential_BINARY_V6/reports/cell17_1_ctgan_baseline_run_audit.csv'
    if not path.is_file():
        return {'audit_available': False, 'all_ctgan_scopes_succeeded': False,
                'unsuccessful_scopes': [], 'reason': 'CTGAN audit is missing.'}
    frame = pd.read_csv(path)
    failed = frame[~frame.status.astype(str).str.lower().eq('success')]
    fields = [x for x in ['scope_id', 'status', 'reason', 'error'] if x in failed]
    return {'audit_available': True, 'all_ctgan_scopes_succeeded': bool(len(frame) and failed.empty),
            'attempted_scopes': len(frame), 'unsuccessful_scopes': failed[fields].fillna('').to_dict('records'),
            'scope': 'CTGAN baseline attempts only; other model audits remain in the reports.'}


def inventory(run):
    run = Path(run)
    items, errors = {}, {}
    roots = [run / 'residential_BINARY_V6/synthetic', run / 'smartstar',
             run / 'toniot', run / 'protocol_regression']
    for root in roots:
        for path in sorted(root.rglob('*.parquet')):
            items[str(path.relative_to(run))] = parquet_digest(path)

    required = [
        'residential_BINARY_V6/synthetic/REAL_TEST_SPLIT.parquet',
        'residential_BINARY_V6/synthetic/CPS_SYNTHETIC_TEST_PUBLIC_Q6_CONSERVATIVE.parquet',
    ]
    for name in required:
        if name not in items:
            errors[name] = 'Required generated output is missing.'

    metrics = run / 'residential_BINARY_V6/reports/cell12c2_all_val_candidate_metrics.csv'
    if not metrics.is_file():
        errors[str(metrics.relative_to(run))] = 'Continuous candidate inventory is missing.'
    else:
        frame = pd.read_csv(metrics, low_memory=False)
        folder = run / 'residential_BINARY_V6/artifacts/cell12c2_candidate_cache_v8_14'
        if frame.empty or frame.candidate_id.duplicated().any():
            errors['continuous_candidates'] = 'Empty inventory or duplicate candidate identifiers.'
        for row in frame.to_dict('records'):
            key = str(row.get('cache_fingerprint_hash', ''))
            path = folder / key[:2] / (key + '.npy')
            name = 'continuous_candidates/' + str(row['candidate_id'])
            if not path.is_file():
                errors[name] = 'Candidate array is missing.'
                continue
            values = np.load(path, allow_pickle=False)
            digest = hashlib.sha256(str(values.dtype).encode() + str(values.shape).encode()
                                    + np.ascontiguousarray(values).tobytes()).hexdigest()
            items[name] = {'sha256_ordered_values': digest, 'rows': len(values),
                           'columns': [str(row.get('col', ''))]}

    tables, excluded_columns, table_errors = {}, {}, {}
    for path in sorted(run.rglob('*.csv')):
        relative = path.relative_to(run)
        selected = (relative.parts[0] in {'protocol_regression', 'window_reconstruction',
                    'controlled_stage', 'smartstar', 'toniot'} or
                    ('residential_BINARY_V6' in relative.parts and
                     ('reports' in relative.parts or 'EXT_OUT' in relative.parts)))
        if not selected:
            continue
        name = str(relative)
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            tables[name] = {'sha256_ordered_values': hashlib.sha256(b'empty-table').hexdigest(),
                            'rows': 0, 'columns': []}
            continue
        except Exception as error:
            table_errors[name] = str(error)
            continue
        # Exclusions are reported per table; generated data retain their own value fingerprints.
        metadata = r'(^|_)(path|sha256|created|created_utc|timestamp|elapsed|runtime|fit_seconds|sample_seconds|started_at|finished_at|mtime|bytes)($|_)|(^hash$|_fingerprint_hash$|_file_hash$)'
        excluded = [column for column in frame if re.search(metadata, column)]
        excluded_columns[name] = excluded
        frame = frame.drop(columns=excluded)
        for column in frame.select_dtypes(include='object'):
            # Object columns may contain booleans and missing values, not text.
            frame[column] = frame[column].map(
                lambda value: value.replace(str(run), '<RUN>') if isinstance(value, str) else value)
        if frame.shape[1] == 0:
            # All fields were explicitly excluded as execution metadata. Retain
            # the row count and exclusion list, but do not claim a value digest.
            tables[name] = {'sha256_ordered_values': None, 'rows': len(frame),
                            'columns': [], 'content_status': 'METADATA_ONLY_EXCLUDED'}
            continue
        digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False, categorize=False)
                                .to_numpy(dtype='<u8').tobytes()).hexdigest()
        tables[name] = {'sha256_ordered_values': digest, 'rows': len(frame),
                        'columns': frame.columns.tolist()}

    inputs = {}
    for dataset in ['smartstar', 'toniot']:
        path = run / dataset / 'input_inventory.json'
        if path.is_file():
            inputs[dataset] = json.loads(path.read_text())
        else:
            errors[dataset + '/input_inventory.json'] = 'Source-file fingerprints are missing.'
    payload = {
        'algorithm': 'SHA-256 of fixed-size ordered pandas row-hash vectors for Parquet/CSV; '
                     'SHA-256 of dtype, shape and bytes for numeric candidate arrays.',
        'outputs': items, 'tables': tables, 'input_inventories': inputs,
        'scope': 'Generated Parquet values, continuous candidate vectors and listed numerical/decision tables. '
                 'Logs, execution timestamps, archive bytes and model serialization are excluded.',
        'metadata_columns_excluded': excluded_columns, 'table_errors': table_errors,
        'inventory_errors': errors, 'baseline_training': baseline_training_status(run),
        'cross_run_verified': False,
    }
    write_json(run / 'repeatability_inventory.json', payload)
    return {'paper_agreement': 'NOT_A_REFERENCE_COMPARISON', 'fingerprinted_outputs': len(items),
            'fingerprinted_tables': len(tables), 'inventory_complete': not errors and not table_errors,
            'baseline_training': payload['baseline_training'], 'exact_repeatability_verified': False}


def read_json(run, name):
    return json.loads((Path(run) / name).read_text())


def compare_runs(left, right, out):
    left, right, out = Path(left), Path(right), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    inventories = [read_json(run, 'repeatability_inventory.json') for run in [left, right]]
    rows = []
    for section in ['outputs', 'tables']:
        a, b = [value[section] for value in inventories]
        for key in sorted(set(a) | set(b)):
            status = ('MISSING' if key not in a or key not in b else
                      'EXACT' if a[key] == b[key] else 'DIFFERENT')
            rows.append({'section': section, 'item': key, 'status': status})
    frame = pd.DataFrame(rows, columns=['section', 'item', 'status'])
    frame.to_csv(out / 'exact_output_comparison.csv', index=False)
    same_environment = read_json(left, 'environment.json') == read_json(right, 'environment.json')
    same_all_packages = (read_json(left, 'environment_full.json')['packages'] ==
                         read_json(right, 'environment_full.json')['packages'])
    same_hardware = read_json(left, 'hardware.json') == read_json(right, 'hardware.json')
    same_package = read_json(left, 'package_identity.json') == read_json(right, 'package_identity.json')
    configs = [{k: v for k, v in read_json(run, 'config.json').items()
                if k not in ['run_dir', 'output_parent']} for run in [left, right]]
    complete = all(read_json(run, 'FINAL_REPORT.json').get('all_stages_completed', False)
                   for run in [left, right])
    fresh = all(read_json(run, 'fresh_continuous_training.json').get('previous_candidate_hits', -1) == 0
                for run in [left, right])
    inventory_ok = all(not value.get('table_errors') and not value.get('inventory_errors')
                       for value in inventories)
    same_inputs = inventories[0].get('input_inventories') == inventories[1].get('input_inventories')
    passed = bool(inventory_ok and len(frame) and frame.status.eq('EXACT').all() and
                  same_environment and same_all_packages and same_hardware and same_package and
                  configs[0] == configs[1] and complete and fresh and same_inputs)
    baselines = [value.get('baseline_training', {}) for value in inventories]
    result = {
        'status': 'PASS_FOR_LISTED_OUTPUTS' if passed else 'DIFFERENT_OR_INCOMPLETE',
        'exact_output_counts': frame.status.value_counts().to_dict(),
        'same_environment': same_environment, 'same_all_recorded_packages': same_all_packages,
        'same_hardware_description': same_hardware, 'same_package': same_package,
        'same_configuration': configs[0] == configs[1], 'same_public_source_files': same_inputs,
        'both_complete': complete, 'both_fresh_continuous_training': fresh,
        'inventories_complete_and_readable': inventory_ok,
        'both_runs_all_ctgan_scopes_succeeded': all(x.get('all_ctgan_scopes_succeeded', False) for x in baselines),
        'baseline_training': {'first': baselines[0], 'second': baselines[1]},
        'universal_repeatability_claim': False,
        'scope': 'See each inventory for exclusions. Matching failed baseline attempts do not establish '
                 'successful training for those scopes. These two executions do not establish agreement '
                 'on other machines or across library versions.',
    }
    write_json(out / 'comparison.json', result)
    return result
