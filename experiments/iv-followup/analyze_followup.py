"""CSV-only analysis, preserving seed pairing. No training code is imported."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def read(path): return list(csv.DictReader(Path(path).open()))
def dump(path,x): Path(path).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(values, seed=20260903):
    x=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    b=x[rng.integers(0,len(x),(5000,len(x)))].mean(axis=1)
    lo,hi=np.quantile(b,[.025,.975])
    return {'mean':float(x.mean()),'low':float(lo),'high':float(hi),'n':len(x),
            'equivalent_002':bool(lo>=-.02 and hi<=.02)}
def fit(x,y):return np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
def predict(x,b):return np.column_stack([np.ones(len(x)),x])@b
def metrics(y,p,base):
    mse=np.mean((y-p)**2);b=np.mean((y-base)**2)
    return {'rmse':float(np.sqrt(mse)),'baseline_rmse':float(np.sqrt(b)),
            'skill_vs_training_mean':float(1-mse/b),
            'pearson_r':float(np.corrcoef(y,p)[0,1]) if np.std(p)>1e-12 else None}


def dataset(folder, legacy=False):
    moment=read(folder/('optimizer_moment_endpoints.csv' if legacy else 'optimizer_moments.csv'))
    mm={(int(r['seed']),r['route'],r['direction']):r for r in moment}
    response={}
    if legacy:
        for r in read(folder/'protocol_summary.csv'):
            if r['protocol']=='combined_switch' and r['optimizer_component']=='own':
                response[(int(r['seed']),r['route'],r['direction'])]=float(r['pulse_drop'])
    else:
        for path in sorted(folder.glob('seed_*/timecourse.csv')):
            start={}
            for r in read(path):
                if r['mode']!='own' or r['trajectory']!='continued_opposition':continue
                key=(int(r['seed']),r['route'],r['direction'])
                if r['step']=='0':start[key]=float(r['aligned_s'])
                if r['step']=='10':response[key]=start[key]-float(r['aligned_s'])
    rows=[]
    for seed in sorted({k[0] for k in mm}):
        row={'seed':seed}
        for metric,name in [('v_sum','v_ratio'),('effective_lr_mean','q_ratio'),('m_l2','m_ratio')]:
            row[name]=np.mean([float(mm[(seed,'acquisition',d)][metric])/float(mm[(seed,'update',d)][metric]) for d in ['A','B']])
        row['response_difference']=np.mean([response[(seed,'acquisition',d)]-response[(seed,'update',d)] for d in ['A','B']])
        row['response_ratio']=np.mean([response[(seed,'acquisition',d)]/response[(seed,'update',d)] for d in ['A','B']])
        rows.append(row)
    return rows


def existing_analysis():
    out=HERE/'existing_analysis';out.mkdir(exist_ok=True)
    old=dataset(HERE/'vendor/experiment_iv_complete_v05/IV-E/results_confirm',True)
    write(out/'seed_predictors.csv',old)
    result={'status':'post-hoc exploratory analysis of existing 40 seeds','models':{}}
    frozen={}
    for outcome in ['response_difference','response_ratio']:
        y=np.array([r[outcome] for r in old]);base=np.array([(y.sum()-v)/(len(y)-1) for v in y])
        for feature in ['q_ratio','v_ratio','m_ratio']:
            x=np.array([r[feature] for r in old]);pred=[]
            for i in range(len(y)):
                keep=np.arange(len(y))!=i;pred.append(predict(x[i:i+1],fit(x[keep],y[keep]))[0])
            key=outcome+'__'+feature
            result['models'][key]=metrics(y,np.array(pred),base)
            frozen[key]={'outcome':outcome,'feature':feature,'coefficients':fit(x,y).tolist(),'training_mean':float(y.mean())}
    q=np.array([r['q_ratio'] for r in old]);y=np.array([r['response_ratio'] for r in old])
    result['unfitted_q_equals_response_ratio']={**metrics(y,q,np.full(len(y),y.mean())),
                                             'q_ratio':ci(q),'response_ratio':ci(y),'ratio_error':ci(y-q)}
    dump(out/'result.json',result);dump(out/'frozen_models.json',frozen)
    print(json.dumps(result,indent=2))


def new_analysis(folder):
    forms=read(folder/'formation_endpoints.csv');by=defaultdict(dict)
    for r in forms:by[(int(r['seed']),r['direction'])][r['route']]=r
    match={}
    for m,tol in [('aligned_s',.02),('target_D',.02),('other_D',.02),('error_sum',.02),('aligned_low_s',.01)]:
        vals=defaultdict(list)
        for (seed,d),r in by.items():vals[seed].append(float(r['acquisition'][m])-float(r['update'][m]))
        c=ci([np.mean(v) for v in vals.values()]);c['margin']=tol;c['pass']=c['low']>=-tol and c['high']<=tol;match[m]=c
    time_rows=[]
    for p in sorted(folder.glob('seed_*/timecourse.csv')):time_rows.extend(read(p))
    cells=defaultdict(dict)
    for r in time_rows:
        key=(int(r['seed']),r['direction'],r['mode'],r['trajectory'],int(r['step']))
        assert r['route'] not in cells[key]
        cells[key][r['route']]=float(r['aligned_s'])
    paired=defaultdict(list);levels=defaultdict(list)
    # First average mirror labels within each seed.
    temp=defaultdict(list); leveltemp=defaultdict(list)
    for (seed,d,m,tr,t),v in cells.items():
        a0=cells[(seed,d,m,tr,0)]['acquisition'];u0=cells[(seed,d,m,tr,0)]['update']
        temp[(seed,m,tr,t)].append((a0-v['acquisition'])-(u0-v['update']))
        for route in ['acquisition','update']:
            leveltemp[(seed,m,tr,t,route)].append(v[route])
    pairedrows=[]
    for (seed,m,tr,t),v in temp.items():
        value=float(np.mean(v));paired[(m,tr,t)].append(value)
        pairedrows.append({'seed':seed,'mode':m,'trajectory':tr,'step':t,'route_drop_difference':value})
    summary=[]
    for (m,tr,t),v in sorted(paired.items()):summary.append({'mode':m,'trajectory':tr,'step':t,**ci(v)})
    levelrows=[]
    for (seed,m,tr,t,route),v in leveltemp.items():levels[(m,tr,t,route)].append(np.mean(v))
    for (m,tr,t,route),v in sorted(levels.items()):levelrows.append({'mode':m,'trajectory':tr,'step':t,'route':route,**ci(v)})
    write(folder/'route_effects.csv',summary);write(folder/'levels.csv',levelrows);write(folder/'seed_route_effects.csv',pairedrows)
    lookup={(r['mode'],r['trajectory'],r['step']):r for r in summary}
    short={m:r for (m,tr,t),r in lookup.items() if tr=='continued_opposition' and t==10}
    differences={}
    byseed={(r['seed'],r['mode']):r['route_drop_difference'] for r in pairedrows if r['step']==10 and r['trajectory']=='continued_opposition'}
    seeds=sorted({r['seed'] for r in pairedrows})
    for a,b in [('own','v_equal_scale'),('v_equal_scale','scalar_v_equal'),('v_equal_scale','v_other_equal_scale'),('own','scalar_v_own'),('own','m_other')]:
        differences[a+'_minus_'+b]=ci([byseed[(s,a)]-byseed[(s,b)] for s in seeds])
    # Level differences at 500/510 are endpoint equivalence, not changes from baseline.
    endpoint={}
    for tr,t in [('continued_opposition',500),('pulse_then_neutral',510)]:
        for mode in ['own','reset']:
            values=[]
            for seed in seeds:
                values.append(np.mean([cells[(seed,d,mode,tr,t)]['acquisition']-cells[(seed,d,mode,tr,t)]['update'] for d in ['A','B']]))
            endpoint[tr+'__'+mode]=ci(values)
    new=dataset(folder);write(folder/'seed_predictors.csv',new)
    frozen=json.loads((HERE/'existing_analysis/frozen_models.json').read_text())
    predictions={}
    for key,model in frozen.items():
        x=np.array([r[model['feature']] for r in new]);y=np.array([r[model['outcome']] for r in new])
        predictions[key]=metrics(y,predict(x,model['coefficients']),np.full(len(y),model['training_mean']))
    q=np.array([r['q_ratio'] for r in new]);y=np.array([r['response_ratio'] for r in new])
    predictions['unfitted_q_equals_response_ratio']={'q_ratio':ci(q),'response_ratio':ci(y),'ratio_error':ci(y-q),
                                                  **metrics(y,q,np.full(len(y),np.mean(y)))}
    result={'n_seeds':len(seeds),'formation_match':match,'short_route_effects':short,
            'between_mode_contrasts':differences,'late_endpoint_differences':endpoint,'external_prediction':predictions,
            'original_pattern_replication':{'own_lower_above_010':short['own']['low']>.1,
                                          'v_swap_negative':short['v_other']['high']<0,
                                          'reset_equivalent':short['reset']['equivalent_002']}}
    dump(folder/'result.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--existing',action='store_true');p.add_argument('--folder',default='results_confirm');a=p.parse_args()
    if a.existing:existing_analysis()
    else:new_analysis(HERE/a.folder)
