"""Reference implementation of the controlled claim-decision experiment.

This code does NOT reproduce the unavailable residential generators, C2ST, Q4,
utility tasks, or historical policy implementation. It tests a new, explicit
binary co-activity benchmark and separately verifies evidence-handling rules.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

VERSION = 'decision-reference-1.0'
POLICY = {
    'version': VERSION,
    'purpose': 'Controlled binary co-activity and marginal-rate preservation only',
    'p': 0.10, 'reference_c': 0.05,
    'candidate_c': [0.05, 0.045, 0.03, 0.0],
    'absolute_probability_tolerance': 0.01,
    'n_rows_per_trajectory': 12000,
    'feature_scopes': [2, 8, 32],
    'trials_per_setting': 100,
    'minimum_rows': 128,
    'low_support_rows': 16,
    'pilot_trials_per_setting': 8,
    'pilot_seed': 319407,
    'final_seed': 928361,
    'calibration': 'Tolerance, sample size, and trial count specified before pilot; pilot measures runtime only',
    'process': 'Rows independent; first pair has P11=c,P10=P01=p-c,P00=1-2p+c; remaining features independent Bernoulli(p)',
    'truth': 'All population marginal rates equal p and abs(candidate_c-reference_c)<=0.01',
    'reduced_evidence_comparators': 'Pooled rate only, or all individual rates; neither measures co-activity',
    'complete_evidence_comparators': 'Same required marginal and co-activity differences, with the same tolerance',
}

@dataclass(frozen=True)
class Claim:
    artifact: str
    scope: str
    policy: str
    required: tuple[str, ...]
    permitted_warnings: tuple[str, ...] = ()
    history_eligible: bool = True
    applicable: bool = True

@dataclass(frozen=True)
class Evidence:
    metric: str
    artifact: str
    scope: str
    policy: str
    status: str = 'evaluated'
    grade: str | None = 'pass'
    value: float | None = 0.0
    blocker: bool = False
    warning_disclosed: bool = False
    support_real: int = 12000
    support_candidate: int = 12000

@dataclass(frozen=True)
class Decision:
    status: str
    reasons: tuple[str, ...]


def assess(claim: Claim, records: list[Evidence], minimum_rows: int = 128) -> Decision:
    """Failing evidence takes precedence; missing evidence never becomes a pass."""
    if not claim.applicable:
        return Decision('not_applicable', ('claim_not_applicable',))
    failed, insufficient, missing = [], [], []
    if not claim.history_eligible:
        missing.append('ineligible_evaluation_history')
    for name in claim.required:
        match = [r for r in records if r.metric == name and
                 (r.artifact, r.scope, r.policy) == (claim.artifact, claim.scope, claim.policy)]
        if len(match) != 1:
            missing.append(f'{name}:missing_or_nonunique_matching_record')
            continue
        r = match[0]
        if r.status == 'insufficient_evidence':
            insufficient.append(f'{name}:insufficient_evidence'); continue
        if r.status != 'evaluated':
            missing.append(f'{name}:{r.status}'); continue
        if r.value is None or not np.isfinite(r.value) or min(r.support_real, r.support_candidate) < minimum_rows:
            insufficient.append(f'{name}:invalid_value_or_low_support'); continue
        if r.grade not in ('pass', 'warning', 'fatal'):
            missing.append(f'{name}:invalid_grade'); continue
        if r.blocker or r.grade == 'fatal':
            failed.append(f'{name}:failed_required_check')
        elif r.grade == 'warning' and (name not in claim.permitted_warnings or not r.warning_disclosed):
            failed.append(f'{name}:warning_not_permitted_and_disclosed')
    reasons = tuple(failed + insufficient + missing)
    if failed: return Decision('withheld', reasons)
    if insufficient: return Decision('insufficient_evidence', reasons)
    if missing: return Decision('not_assessed', reasons)
    if not claim.required: return Decision('not_assessed', ('empty_required_set',))
    return Decision('supported', ('all_declared_requirements_met',))


def truth_table_oracle(claim: Claim, records: list[Evidence], minimum_rows: int = 128) -> str:
    """Independent Boolean specification; does not call assess or its helpers."""
    if not claim.applicable: return 'not_applicable'
    by_key = {}
    for r in records:
        by_key.setdefault((r.metric, r.artifact, r.scope, r.policy), []).append(r)
    flags = []
    for name in claim.required:
        row = by_key.get((name, claim.artifact, claim.scope, claim.policy), [])
        if len(row) != 1:
            flags.append('M'); continue
        r = row[0]
        if r.status == 'insufficient_evidence': flags.append('I'); continue
        if r.status != 'evaluated': flags.append('M'); continue
        if r.value is None or not np.isfinite(r.value) or r.support_real < minimum_rows or r.support_candidate < minimum_rows:
            flags.append('I'); continue
        if r.grade not in ['pass','warning','fatal']: flags.append('M'); continue
        row_ok = (not r.blocker) and ((r.grade == 'pass') or
                  (r.grade == 'warning' and r.warning_disclosed and name in claim.permitted_warnings))
        flags.append('P' if row_ok else 'F')
    if 'F' in flags: return 'withheld'
    if 'I' in flags: return 'insufficient_evidence'
    if ('M' in flags) or not claim.history_eligible or not flags: return 'not_assessed'
    return 'supported'


def policy_tests():
    claim = Claim('candidate-v1', 'scope-v1', VERSION, ('marginal','joint'), ('joint',))
    first = Evidence('marginal',claim.artifact,claim.scope,claim.policy)
    second = Evidence('joint',claim.artifact,claim.scope,claim.policy)
    variants = {
        'pass': second,
        'fatal': replace(second,grade='fatal'),
        'explicit_blocker': replace(second,blocker=True),
        'permitted_warning': replace(second,grade='warning',warning_disclosed=True),
        'undisclosed_warning': replace(second,grade='warning'),
        'insufficient_status': replace(second,status='insufficient_evidence',grade=None,value=None),
        'not_assessed': replace(second,status='not_assessed',grade=None,value=None),
        'required_not_applicable': replace(second,status='not_applicable',grade=None,value=None),
        'low_support': replace(second,support_candidate=2),
        'nan': replace(second,value=float('nan')),
        'wrong_artifact': replace(second,artifact='candidate-v0'),
        'wrong_scope': replace(second,scope='scope-v0'),
        'wrong_policy': replace(second,policy='different'),
        'invalid_grade': replace(second,grade='unknown'),
    }
    rows=[]
    def check(name,c,rr,expected=None):
        a=assess(c,rr).status; b=truth_table_oracle(c,rr)
        assert a==b, (name,a,b)
        if expected is not None: assert a==expected,(name,a,expected)
        rows.append({'test':name,'decision':a,'oracle':b,'passed':True})
    # Explicit hand-written expectations are separate from the two algorithms.
    check('identity_support',claim,[first,second],'supported')
    check('fatal_with_missing',claim,[replace(first,grade='fatal')],'withheld')
    check('missing_required',claim,[first],'not_assessed')
    check('duplicate_required',claim,[first,second,second],'not_assessed')
    check('warning_not_allowed',replace(claim,permitted_warnings=()),[first,variants['permitted_warning']],'withheld')
    check('adaptive_confirmatory_history',replace(claim,history_eligible=False),[first,second],'not_assessed')
    check('whole_claim_inapplicable',replace(claim,applicable=False),[],'not_applicable')
    check('fixed_scope_blocker_cannot_be_overridden',claim,[first,variants['explicit_blocker'],Evidence('extra','candidate-v1','scope-v1',VERSION)],'withheld')
    narrow=Claim('candidate-v1','narrow-v1',VERSION,('marginal',))
    check('changed_scope_requires_matching_evidence',narrow,[first,second],'not_assessed')
    check('changed_scope_separate_claim',narrow,[replace(first,scope='narrow-v1')],'supported')
    for key,row in variants.items(): check(key,claim,[first,row])
    for (a,x),(b,y) in itertools.product(variants.items(),repeat=2):
        check(f'pair_{a}_{b}',claim,[replace(x,metric='marginal'),y])
    return rows


def draw_process(rng,n,m,c):
    p=POLICY['p']
    cat=rng.choice(4,size=n,p=[1-2*p+c,p-c,p-c,c])
    x=np.empty((n,m),dtype=np.uint8)
    x[:,0]=(cat==1)|(cat==3);x[:,1]=(cat==2)|(cat==3)
    if m>2:x[:,2:]=(rng.random((n,m-2))<p)
    return x


def measure(reference,candidate):
    rr=reference.mean(axis=0);ss=candidate.mean(axis=0)
    return {'pooled_rate_error':float(abs(rr.mean()-ss.mean())),
            'marginal_errors':abs(rr-ss),
            'joint_error':float(abs(np.mean(reference[:,0]*reference[:,1])-np.mean(candidate[:,0]*candidate[:,1])))}


def run_one(seed,c,m,n,phase,trial):
    # Spawn independently for the reference and candidate within every trial.
    a,b=np.random.SeedSequence(seed).spawn(2)
    ref=draw_process(np.random.default_rng(a),n,m,POLICY['reference_c'])
    syn=draw_process(np.random.default_rng(b),n,m,c)
    v=measure(ref,syn);tol=POLICY['absolute_probability_tolerance']
    known_ok=abs(c-POLICY['reference_c'])<=tol+1e-12
    scope=f'binary-{m}'
    claim=Claim(f'{phase}-{m}-{c}-{trial}',scope,VERSION,tuple([f'marginal-{j}' for j in range(m)]+['joint-01']))
    values=list(v['marginal_errors'])+[v['joint_error']]
    records=[Evidence(name,claim.artifact,scope,VERSION,
                      status='evaluated' if n>=POLICY['minimum_rows'] else 'insufficient_evidence',
                      grade=(None if n<POLICY['minimum_rows'] else ('pass' if value<=tol else 'fatal')),value=float(value),support_real=n,support_candidate=n)
             for name,value in zip(claim.required,values)]
    core=assess(claim,records,POLICY['minimum_rows']).status
    oracle=truth_table_oracle(claim,records,POLICY['minimum_rows'])
    assert core==oracle
    complete='insufficient_evidence' if n<POLICY['minimum_rows'] else ('supported' if all(z<=tol for z in values) else 'withheld')
    assert complete==core
    results={
      'pooled':v['pooled_rate_error']<=tol,
      'per_feature':bool(np.all(v['marginal_errors']<=tol)),
    }
    decisions={k:('insufficient_evidence' if n<POLICY['minimum_rows'] else ('supported' if v else 'withheld')) for k,v in results.items()}
    decisions.update(same_evidence=complete,framework=core)
    return [{'phase':phase,'trial':trial,'seed':int(seed),'features':m,'n_rows':n,'candidate_c':c,
             'true_union_rate':0.20-c,'truth':'acceptable' if known_ok else 'unacceptable',
             'method':method,'decision':dec,'pooled_rate_error':v['pooled_rate_error'],
             'max_marginal_error':float(max(v['marginal_errors'])),'joint_error':v['joint_error']}
            for method,dec in decisions.items()]


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(rows):
    keys=('phase','features','n_rows','method','truth')
    groups={}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for key,rr in groups.items():
        n=len(rr);d={**dict(zip(keys,key)),'trials':n}
        for state in ['supported','withheld','insufficient_evidence','not_assessed']:
            d[state]=sum(r['decision']==state for r in rr)
        out.append(d)
    return out


def mathematical_tests():
    rows=[]
    def add(name,actual,expected,note):
        assert np.isclose(actual,expected,rtol=1e-12,atol=1e-12),(name,actual,expected)
        rows.append({'test':name,'actual':float(actual),'expected':float(expected),'passed':True,'meaning':note})
    # Binary identities from exact probability mass functions.
    p,q=.001,.002
    add('binary_KS',max(abs(np.array([1-p,1])-np.array([1-q,1]))),abs(p-q),'Population binary KS equals rate difference')
    add('binary_W1',abs((1-p)-(1-q)),abs(p-q),'Unit spacing; not a second independent difference')
    for m,expected in [(0,1.0),(3,0.524893534183932),(10,0.500022699964881)]:
        eta=.5+.5*np.exp(-m)
        add(f'ETA_rho1_m{m}',eta,expected,'Analytical score property only; complete historical Q4 gate unverified')
        assert eta>=.5
    add('folded_AUC_pair',np.mean([max(.48,1-.48),max(.52,1-.52)]),.52,'Finite-sample folding changes null mean')
    # Joint patterns share rates and union but not pairwise co-occurrence.
    A=np.array([[0,0,0]]*2+[[1,0,0]]*2+[[0,1,1]]*4)
    B=np.array([[0,0,0]]*2+[[0,1,1]]*2+[[1,1,0]]*2+[[0,0,1]]*2)
    add('equal_marginals',max(abs(A.mean(0)-B.mean(0))),0,'Marginals do not identify co-occurrence')
    add('equal_union',abs(np.any(A,1).mean()-np.any(B,1).mean()),0,'Union alone does not identify co-occurrence')
    add('different_pair01',abs(np.mean(A[:,0]*A[:,1])-np.mean(B[:,0]*B[:,1])),.25,'Counterexample to full joint realism from union')
    # Own-observed and joint-observed populations differ even with equal mask rates.
    real=np.array([0,0,10,10]);syn=np.array([0,0,10,10])
    mr=np.array([1,0,1,0],dtype=bool);ms=np.array([1,1,0,0],dtype=bool)
    both=mr&ms
    add('joint_observed_value_difference',abs(real[both].mean()-syn[both].mean()),0,'Conditional equality only')
    add('own_observed_mean_difference',abs(real[mr].mean()-syn[ms].mean()),5,'Different observable populations, not imputation')
    add('same_mask_rate',abs(mr.mean()-ms.mean()),0,'Rate agreement alone does not prove reporting/value dependence')
    add('real_overlap_fraction',both.sum()/mr.sum(),.5,'Half of each observed population omitted by intersection')
    # Hand-check time semantics. Missing rows cannot be compressed for clock lags.
    times=np.arange(5);observed=np.array([1,0,1,0,1],dtype=bool)
    add('clock_lag1_valid_pairs',sum(observed[:-1]&observed[1:]),0,'No valid adjacent-time pairs despite three observed values')
    reports=np.array([0,1,0,1,0],dtype=bool);last=None;stale=[]
    for t,reported in enumerate(reports):
        if reported:last=t
        stale.append(np.nan if last is None else t-last)
    assert np.isnan(stale[0]) and stale[1:]==[0,1,0,1]
    add('staleness_reset',stale[3],0,'Initialize unknown when no prior report is available')
    value,physically_active,is_observed=0,False,True
    preserved=value if is_observed else np.nan
    add('observed_OFF_is_valid',preserved,0,'Physical inactivity is not missing observation')
    return rows


def run(output,overwrite=False):
    output=Path(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f'{output} is nonempty; choose a new run directory')
    output.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    protocol=json.dumps(POLICY,sort_keys=True,indent=2)
    # Freeze before pilot or final draws; the digest shows identity, not non-exposure.
    (output/'protocol_frozen.json').write_text(protocol+'\n')
    protocol_digest=sha256(protocol.encode()).hexdigest()
    tests=policy_tests();maths=mathematical_tests()
    write_csv(output/'policy_tests.csv',tests);write_csv(output/'analytical_checks.csv',maths)
    pilot=[]
    seed_gen=np.random.default_rng(POLICY['pilot_seed'])
    for c in POLICY['candidate_c']:
        for t in range(POLICY['pilot_trials_per_setting']):
            seed=int(seed_gen.integers(0,2**62));pilot.extend(run_one(seed,c,2,POLICY['n_rows_per_trajectory'],'pilot',t))
    write_csv(output/'pilot.csv',pilot)
    pilot_seconds=time.perf_counter()-t0
    final=[]
    seed_gen=np.random.default_rng(POLICY['final_seed'])
    for m in POLICY['feature_scopes']:
        for c in POLICY['candidate_c']:
            for t in range(POLICY['trials_per_setting']):
                seed=int(seed_gen.integers(0,2**62))
                final.extend(run_one(seed,c,m,POLICY['n_rows_per_trajectory'],'final',t))
    for c in POLICY['candidate_c']:
        for t in range(POLICY['trials_per_setting']):
            seed=int(seed_gen.integers(0,2**62))
            final.extend(run_one(seed,c,2,POLICY['low_support_rows'],'low_support',t))
    write_csv(output/'trial_results.csv',final)
    summary=summarize(final);write_csv(output/'truth_by_decision.csv',summary)
    detail=[]
    for m in POLICY['feature_scopes']:
        for c in POLICY['candidate_c']:
            for method in ['pooled','per_feature','same_evidence','framework']:
                rr=[r for r in final if r['phase']=='final' and r['features']==m and r['candidate_c']==c and r['method']==method]
                detail.append({'features':m,'candidate_c':c,'truth':rr[0]['truth'],'method':method,'trials':len(rr),
                               'supported':sum(r['decision']=='supported' for r in rr),
                               'withheld':sum(r['decision']=='withheld' for r in rr),
                               'insufficient_evidence':sum(r['decision']=='insufficient_evidence' for r in rr)})
    write_csv(output/'decision_by_setting.csv',detail)
    meta={'version':VERSION,'completed_utc':datetime.now(timezone.utc).isoformat(),
          'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
          'protocol_sha256':protocol_digest,'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
          'pilot_seconds':pilot_seconds,'total_seconds':time.perf_counter()-t0,
          'policy_tests':len(tests),'analytical_tests':len(maths),'independent_final_trial_pairs':len(final)//4,
          'adequate_support_trial_pairs':1200,'low_support_trial_pairs':400,
          'same_evidence_agreement':True,'private_data_used':False,
          'limitation':'Controlled experiment only; residential pipeline not reproduced'}
    (output/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    files=[{'file':f.name,'sha256':sha256(f.read_bytes()).hexdigest()} for f in sorted(output.iterdir()) if f.is_file()]
    write_csv(output/'SHA256SUMS.csv',files)
    return meta,summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='controlled_results')
    args=parser.parse_args();meta,summary=run(args.output)
    print(json.dumps(meta,indent=2))
    for row in summary:
        if row['features']==2: print(row)
