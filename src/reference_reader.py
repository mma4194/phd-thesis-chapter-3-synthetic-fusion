#!/usr/bin/env python3
"""Read-only comparison of saved outputs with a fixed, versioned paper reference."""
import argparse, datetime, hashlib, json, math, sys, zipfile
from pathlib import Path
import pandas as pd


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def compare(actual,expected,decimals):
    aa=actual if isinstance(actual,list) else [actual]
    ee=expected if isinstance(expected,list) else [expected]
    dd=decimals if isinstance(decimals,list) else [decimals]*len(ee)
    if len(aa)!=len(ee) or len(dd)!=len(ee):raise ValueError('Shape mismatch')
    aa=[float(x) for x in aa];ee=[float(x) for x in ee]
    if not all(math.isfinite(x) for x in aa+ee):raise ValueError('Nonfinite numeric evidence')
    display=all(f'{a:.{d}f}'==f'{e:.{d}f}' for a,e,d in zip(aa,ee,dd))
    strict=all(math.isclose(a,e,rel_tol=1e-7,abs_tol=1e-9) for a,e in zip(aa,ee))
    return display,strict

class Reader:
    def __init__(self,reference,run,evaluation):
        self.spec={x['id']:x for x in reference['checks']};self.roots={'run':run,'evaluation':evaluation}
        self.values={};self.tables={};self.fingerprints={};self.busy=set()
    def path(self,s):
        root=self.roots[s.get('root','run')].resolve();p=(root/s['path']).resolve()
        if not p.is_relative_to(root):raise ValueError('Source escapes configured root')
        if not p.is_file():raise FileNotFoundError(str(p))
        return p
    def csv(self,s):
        p=self.path(s)
        if p not in self.tables:
            self.fingerprints[str(p)]={'kind':'file_sha256','sha256':digest(p)}
            self.tables[p]=pd.read_csv(p)
        d=self.tables[p]
        for k,v in s.get('where',{}).items():
            if isinstance(v,(int,float)) and not isinstance(v,bool):d=d.loc[pd.to_numeric(d[k],errors='coerce').eq(v)]
            else:d=d.loc[d[k].astype(str).eq(str(v))]
        return d
    def get(self,k):
        if k in self.values:return self.values[k]
        if k in self.busy:raise ValueError('Cyclic dependency')
        self.busy.add(k)
        try:
            s=self.spec[k];op=s['op']
            if op in ['identity','subtract']:
                v=[self.get(x) for x in s['dependencies']];value=v[0] if op=='identity' else v[0]-v[1]
            elif op in ['csv','csv_pair','keep_count']:
                d=self.csv(s)
                if op=='keep_count':value=int(d.release_decision.astype(str).str.startswith('keep').sum())
                elif op=='csv' and s['aggregate']=='rows':value=len(d)
                else:
                    col=pd.to_numeric(d[s['column']],errors='raise')
                    if len(col)==0:raise ValueError('Empty selected scope')
                    if op=='csv_pair':value=[float(col.mean()),float(col.std(ddof=1))]
                    elif s['aggregate']=='single':
                        if len(col)!=1:raise ValueError(f'Expected one row, got {len(col)}')
                        value=float(col.iloc[0])
                    else:value=float(getattr(col,s['aggregate'])())
            elif op=='json':
                p=self.path(s);self.fingerprints[str(p)]={'kind':'file_sha256','sha256':digest(p)}
                value=json.loads(p.read_text())[s['key']]
            elif op=='parquet':
                import pyarrow.parquet as pq
                p=self.path(s);f=pq.ParquetFile(p);names=f.schema_arrow.names
                metadata={'rows':f.metadata.num_rows,'physical_columns':names,'size_bytes':p.stat().st_size}
                self.fingerprints[str(p)]={'kind':'metadata_only_NOT_content_hash',**metadata}
                metric=s['metric']
                if metric=='rows':value=f.metadata.num_rows
                elif metric=='columns':value=len(names) if k=='PublicColsOnDisk' else len([n for n in names if not n.startswith('__index')])
                elif metric=='prefix':value=sum(n.startswith(s['value']) for n in names)
                elif metric=='split':
                    table=f.read(columns=['split']).to_pandas();value=int(table['split'].eq(s['value']).sum())
                else:raise ValueError(metric)
            else:raise ValueError('Unknown operation: '+op)
            self.values[k]=value;return value
        finally:self.busy.remove(k)

