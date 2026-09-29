def load_existing_roles(run_root, embedded):
    records={r['feature_name']:dict(r) for r in embedded};paths=[]
    p=Path(run_root)/'artifacts/iot_role_ownership.csv'
    if p.is_file():
        d=pd.read_csv(p,dtype=str,keep_default_na=False)
        if not {'col','primary_owner'}.issubset(d):raise ValueError('Unexpected iot_role_ownership.csv columns.')
        if d.col.duplicated().any():raise ValueError('Duplicate role-manifest features.')
        for r in d.to_dict('records'):
            prior=records.get(r['col'],{})
            records[r['col']]={**prior,'feature_name':r['col'],'pipeline_owner':r['primary_owner'],'reason':r.get('reason',''),'target_cell':r.get('target_cell','')}
        paths.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'kind':'early_pipeline_role_manifest_not_final_ground_truth'})
    routing=Path(run_root)/'reports/cell12a_v24_1_cont_routing_policy.csv'
    if routing.is_file():
        d=pd.read_csv(routing,dtype=str,keep_default_na=False)
        if 'col' not in d or d.col.duplicated().any():raise ValueError('Invalid continuous routing-policy manifest.')
        for r in d.to_dict('records'):
            prior=records.get(r['col'],{})
            records[r['col']]={**prior,**{k:r.get(k,'') for k in ['family','synthesis_subfamily','policy_bucket','expected_stage12c_kind']}}
        paths.append({'path':str(routing),'sha256':hashlib.sha256(routing.read_bytes()).hexdigest(),'kind':'later_continuous_routing_policy_not_a_recomputed_role'})
    return records,paths


def write_html(path, summaries, details, samples, windows, context):
    bysample={};bywindow={}
    for x in samples:bysample.setdefault(x['feature_name'],[]).append(x)
    for x in windows:bywindow.setdefault(x['feature_name'],[]).append(x)
    esc=lambda x:html.escape(str(x))
    pieces=['<!doctype html><html><head><meta charset="utf-8"><title>IoT feature inspection</title><style>body{font:15px system-ui;margin:24px;max-width:1500px;color:#172b4d;background:#f7f9fc}header{position:sticky;top:0;background:#f7f9fc;padding:12px 0}input{width:65%;padding:10px}details{background:white;margin:10px 0;padding:14px;border:1px solid #ccd5e0;border-radius:6px}summary{cursor:pointer;font-weight:650;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #dde3ea;padding:5px;text-align:left;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}.badge{font-weight:400;color:#446}button{padding:8px;margin:5px}</style></head><body>',
            '<header><h1>Feature values and representation review</h1><p>'+esc(context)+'</p><input id="search" placeholder="Filter by feature, blind ID, representation or semantic hint"><button onclick="document.querySelectorAll(\'details\').forEach(x=>x.open=false)">Collapse all</button><span id="count"></span></header>',
            '<p>Inspection only. Existing assignments and name hints are not adjudicated ground truth. Samples retain stored precision; rows may be temporally dependent.</p>']
    for s,d in zip(summaries,details):
        name=s['feature_name'];label=name+' '+str(s['blind_row_id'])+' '+s['observed_representation']+' '+s['semantic_name_hint']
        pieces.append('<details data-search="'+esc(label.lower())+'"><summary>'+esc(name)+' <span class="badge">'+esc(s['blind_row_id'])+' · '+esc(s['observed_representation'])+'</span></summary>')
        pieces.append('<pre>'+esc(json.dumps(d,indent=2,default=str))+'</pre><h3>Random samples and extremes</h3>')
        pieces.append(pd.DataFrame(bysample.get(name,[])).drop(columns=['feature_name'],errors='ignore').to_html(index=False,escape=True))
        if name in bywindow:
            pieces.append('<h3>Temporal context: targeted, not random prevalence estimates</h3>'+pd.DataFrame(bywindow[name]).drop(columns=['feature_name']).to_html(index=False,escape=True))
        pieces.append('</details>')
    pieces.append('<script>const q=document.getElementById("search");function filter(){let n=0;document.querySelectorAll("details").forEach(d=>{d.hidden=!d.dataset.search.includes(q.value.toLowerCase());if(!d.hidden)n++});document.getElementById("count").textContent=" "+n+" features"}q.addEventListener("input",filter);filter();</script></body></html>')
    path.write_text(''.join(pieces),encoding='utf-8')


