"""Publication figures from computed run outputs; no reference-value substitution."""
from pathlib import Path
import csv,json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from workflow import ROOT,write_json

def generate(run):
    run=Path(run);out=run/'figures';out.mkdir(exist_ok=True)
    source=(ROOT/'public/src/plot_figures.py').read_text()
    style=source[source.index("blue='#003FA8'"):source.index('# Controlled benchmark:')]
    first=source[source.index('# Controlled benchmark:'):source.index('# Saved residential')]
    second=source[source.index('def macro(name):'):]
    first=first.replace("fig.savefig(out/'fig02_controlled_decisions.pdf');","fig.savefig(out/'fig02_controlled_decisions.pdf');fig.savefig(out/'fig02_controlled_decisions.png',dpi=180);")
    second=second.replace("fig.savefig(out/'fig03_empirical_quality.pdf');","fig.savefig(out/'fig03_empirical_quality.pdf');fig.savefig(out/'fig03_empirical_quality.png',dpi=180);")
    namespace={'Path':Path,'csv':csv,'json':json,'math':math,'np':np,'matplotlib':matplotlib,'plt':plt,'out':out}
    exec(style,namespace);report=[]
    controlled=run/'controlled_stage/controlled'
    if (controlled/'decision_by_setting.csv').exists():
        namespace['controlled_dir']=controlled;exec(first,namespace)
        report.append({'figure':'Figure 2','status':'GENERATED','source':str(controlled/'decision_by_setting.csv')})
    else:report.append({'figure':'Figure 2','status':'MISSING_NEW_RESULT'})
    active=run/'residential_BINARY_V6/reports/thesis_strong_revision_fixes/tables/q6_sparse_driver_sequence_risk_null_calibration.csv'
    grades=run/'smartstar/ledgers/test_evidence_q1_q2_q3_ledger.csv'
    if active.exists() and grades.exists():
        d=pd.read_csv(active).set_index('comparison');g=pd.read_csv(grades)
        values={k:float(d.loc[label,'query_active_union_rate']) for k,label in [('QSixSeqRealUnionRate','real_full_vs_real_full_self_description'),('QSixSeqRealHoldoutUnionRate','real_holdout_vs_real_reference_null'),('QSixSeqSyntheticUnionRate','synthetic_vs_real_reference_calibrated')]}
        for title,role in [('Continuous','iot_continuous'),('Binary','iot_binary'),('Driver','iot_driver'),('Obs','iot_observability')]:
            for status in ['Pass','Warning','Fatal']:values['SmartStar'+title+status]=int((g.owner_role.eq(role)&g.test_status.eq(status.lower())).sum())
        namespace['values']=values
        # Preserve the established publication style while changing its data provenance.
        second=second.replace("'results.tex:'+key","str(active)+':'+key").replace("'results.tex:SmartStar'","str(grades)+':SmartStar'")
        namespace.update(active=active,grades=grades);exec(second,namespace)
        report.append({'figure':'Figure 3','status':'GENERATED','source':[str(active),str(grades)]})
    else:report.append({'figure':'Figure 3','status':'MISSING_NEW_RESULT','required':[str(active),str(grades)]})
    # The supplement's filtering counts are measured from the generated tables.
    import pyarrow.parquet as pq
    syn=run/'residential_BINARY_V6/synthetic'
    paths=[syn/'CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet',syn/'CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet',syn/'CPS_SYNTHETIC_TEST_PUBLIC_Q6_CONSERVATIVE.parquet']
    if all(p.is_file() for p in paths):
        counts=[len([c for c in pq.ParquetFile(p).schema_arrow.names if not c.startswith('__index')]) for p in paths]
        labels=['Full scientific table','Role and grade filter','Active-window filter']
        fig,ax=plt.subplots(figsize=(6.7,2.5));bars=ax.barh(range(3),counts,color=['#9FB5D3','#4C78A8','#003FA8']);ax.invert_yaxis();ax.set_yticks(range(3),labels);ax.set_xlabel('Logical columns retained');ax.set_xlim(0,max(counts)*1.12)
        for y,n in enumerate(counts):ax.text(n+max(counts)*.01,y,str(n),va='center',color='black')
        fig.tight_layout();fig.savefig(out/'figS1_candidate_filtering.pdf',bbox_inches='tight');fig.savefig(out/'figS1_candidate_filtering.png',bbox_inches='tight',dpi=180);plt.close(fig)
        pd.DataFrame({'stage':labels,'columns':counts,'source':list(map(str,paths))}).to_csv(out/'figS1_data.csv',index=False)
        report.append({'figure':'Figure S1','status':'GENERATED','source':list(map(str,paths))})
    else:report.append({'figure':'Figure S1','status':'MISSING_NEW_RESULT'})
    write_json(out/'figure_provenance.json',report)
    return {'paper_agreement':'FIGURE_DATA_FROM_NEW_OUTPUTS','figures':report,'all_numerical_figures_generated':all(r['status']=='GENERATED' for r in report),'note':'Figure 1 is an authored workflow diagram, not an experimental output.'}
