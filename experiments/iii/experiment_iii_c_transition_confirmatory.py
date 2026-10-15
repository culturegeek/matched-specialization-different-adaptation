#!/usr/bin/env python3
"""実験III-C.1: 有限転換域だけを未使用seedで再確認する。"""
from __future__ import annotations
import argparse,copy,csv,json,platform,sys
from pathlib import Path
import numpy as np
from experiment_iii_a0 import sha256
from pilot_a2_capacity_phases import LinearSharedModel,Settings
from pilot_c_multivariable_loop import data,step_loop,HISTORY,COMMON

SEEDS=tuple(range(4200,4240));KS=(0,1600,3200,4000,4800,5600,6400);RECOVERY=4000
GATES="""
K0 switch rate=0; K1600<=0.05; K6400>=0.90; rates monotone over 1600--6400;
K6400-K1600>=0.85. Unit=80 mirrored arms from seed 4200--4239.
"""
def args():p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',default='results_c_transition_confirmatory');return p.parse_args()
def main():
 a=args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);cfg=Settings();rows=[]
 for seed in SEEDS:
  train=data(seed,cfg.train_n,.61);test=data(seed+100000,cfg.test_n,.61)
  for target,sign in [('d',1),('h',-1)]:
   m=LinearSharedModel(np.random.default_rng(seed+400000));k=v=float(sign)
   k,v,*_=step_loop(m,train,test,HISTORY,seed+1,cfg,k,v,target);k,v,dd,dh,s,p=step_loop(m,train,test,COMMON,seed+2,cfg,k,v,None);ps=1 if s>0 else -1
   for K in KS:
    mp=copy.deepcopy(m);kp,vp=k,v;opp='h' if ps>0 else 'd'
    if K:kp,vp,dd0,dh0,s0,p0=step_loop(mp,train,test,K,seed+100+K,cfg,kp,vp,opp)
    else:dd0,dh0,s0,p0=dd,dh,s,p
    kp,vp,ddr,dhr,sr,pr=step_loop(mp,train,test,RECOVERY,seed+500+K,cfg,kp,vp,None)
    rows.append({'seed':seed,'history':target,'K':K,'aligned_perturb_s':s0*ps,'aligned_kappa':kp*ps,'aligned_end_s':sr*ps,'switched':int(sr*ps<0)})
 with (out/'arms.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 rates={K:float(np.mean([r['switched'] for r in rows if r['K']==K])) for K in KS};ordered=[rates[k] for k in KS[1:]]
 gate= rates[0]==0 and rates[1600]<=.05 and rates[6400]>=.90 and all(x<=y for x,y in zip(ordered,ordered[1:])) and rates[6400]-rates[1600]>=.85
 gates={'G3_finite_transition_reconfirmation':{'pass':gate,'rates':rates,'strong_weak_difference':rates[6400]-rates[1600]}}
 comp={'rows':len(rows),'expected':len(SEEDS)*2*len(KS),'unique':len({(r['seed'],r['history'],r['K']) for r in rows})};comp['pass']=comp['rows']==comp['expected']==comp['unique']
 if not comp['pass']:raise RuntimeError(str(comp))
 result={'experiment':'III-C.1','seeds':list(SEEDS),'preregistered_gates':GATES.strip(),'gates':gates,'completeness':comp}
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# 実験III-C.1 結果：有限転換域の独立再確認','',f"- **G3_finite_transition_reconfirmation**: {'PASS' if gate else 'FAIL'}",'','|K|転換率|','|---:|---:|']+[f'|{k}|{rates[k]:.3f}|' for k in KS]
 lines += ['','III-CのK=3200判定不成立は維持する。本実験は、より弱いK=1600と強いK=6400の間に有限転換域があるかを独立seedで確認したものである。']
 (out/'result_c_transition_confirmatory.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 prov={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'script_sha256':sha256(Path(__file__).resolve()),'preregistration_sha256':sha256(Path(__file__).parent/'preregistration_c_transition_confirmatory.md')}
 (out/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'gates':{'G3_finite_transition_reconfirmation':gate},'completeness':comp,'output':str(out)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
