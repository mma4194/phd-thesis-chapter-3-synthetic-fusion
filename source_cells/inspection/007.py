# Inspection utilities: no imports or execution of the original synthesis notebook.
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, html, re, math, gc, uuid
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype


def stable_rng(feature, purpose, seed):
    token = hashlib.sha256(f'{seed}|{feature}|{purpose}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(token[:8], 'little'))


def show_value(value):
    """Exact stored scalar representation; no six-decimal rounding or float32 cast."""
    if value is None or value is pd.NA or value is pd.NaT:
        return '<NULL>'
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value): return '<NaN>'
        if math.isinf(value): return '+Infinity' if value > 0 else '-Infinity'
        return repr(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return '[' + ', '.join(show_value(v) for v in value) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join(repr(k)+': '+show_value(v) for k,v in value.items()) + '}'
    return repr(value)


def read_interval(pf, columns, start, stop, batch_rows=65536):
    """Column projection and row slicing BEFORE conversion to pandas.
    May decompress a Parquet column chunk that overlaps the boundary, but only
    selected rows enter any statistics, samples, timestamps or output.
    """
    if not 0 <= start < stop <= pf.metadata.num_rows:
        raise ValueError('Invalid inspection interval.')
    chunks=[]; offset=0
    for batch in pf.iter_batches(columns=list(columns), batch_size=batch_rows, use_threads=True):
        end=offset+batch.num_rows
        lo,hi=max(start,offset),min(stop,end)
        if hi>lo: chunks.append(batch.slice(lo-offset,hi-lo))
        offset=end
        if offset>=stop: break
    if not chunks: raise ValueError('No rows read.')
    table=pa.Table.from_batches(chunks)
    # Arrow-backed pandas preserves nullable integers without float conversion.
    frame=table.to_pandas(types_mapper=pd.ArrowDtype)
    if len(frame)!=stop-start: raise RuntimeError('Unexpected interval length.')
    return frame


def semantic_hint(name):
    s=name.lower()
    if name in {'sec','sec_epoch_s__canon'} or s.startswith('provenance__'):
        return 'time_or_provenance'
    if s.startswith(('iot__entity_obs__','iot__entity_stale__')) or s in {'iot__tier_present','iot__obs_present','iot__any_update_raw'} or re.search(r'__(obs_present|traffic_present|mask|staleness_s|stale_flag)$',s):
        return 'observability_or_availability'
    if s.startswith('events_in_sec__'): return 'explicit_event_namespace'
    if s.startswith('telemetry_in_sec__'): return 'telemetry_reporting_helper'
    if s.startswith(('router__','zigbee__','zwave__','ota__','ota24__','ota5__','net__','zb__')):
        return 'network_or_wireless_namespace'
    if any(t in s for t in ['linkquality','__rssi','__lqi','signal_strength']):
        return 'wireless_metric_in_telemetry_namespace'
    if s.startswith('iot__events_'): return 'global_event_or_reporting_aggregate_definition_needed'
    if 'trigger_count' in s: return 'trigger_counter_definition_needed'
    if 'rgb_color' in s or 'rgbw_color' in s: return 'colour_representation_check'
    if any(t in s for t in ['count','coffees','cups','consumption']): return 'count_or_accumulator_name'
    if '__number__' in s: return 'numeric_setting_name'
    if s.startswith('iot__'): return 'device_or_sensor_telemetry'
    return 'unclassified_namespace'


