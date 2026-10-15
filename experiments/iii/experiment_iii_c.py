#!/usr/bin/env python3
"""実験III-C: 多変数閉ループ、時間尺度分離、秩序変数候補の確認実験。"""
from __future__ import annotations
import argparse,copy,csv,json,platform,sys
from pathlib import Path
import numpy as np
from experiment_iii_a0 import sha256
from pilot_a2_capacity_phases import LinearSharedModel,Settings
from pilot_c_multivariable_loop import data,step_loop,HISTORY,COMMON,SPONTANEOUS,CAPACITY

SEEDS=tuple(range(4100,4140));KS=(0,1600,3200,4000,4800,5600,6400);RECOVERY=4000
BOOT_SEED=20260824;BOOT_N=10000
GATES="""
G1 branches: correct history branch 80/80; common-end |s| CI lower >0.90.
G2 timescales: spontaneous switches 0 over 8000; K1600 return rate >=0.90;
   90th-percentile relaxation <=500; residence lower-bound / relaxation >=16.
G3 transition: K3200--6400 switch rates monotone; K3200<=0.10; K6400>=0.90; difference>=0.80.
G4 q: K4000/4800/5600 mean aligned-s range <0.01 and switch-rate range>=0.40;
   AUC(-aligned_kappa)>=0.70 and exceeds AUC(-aligned_s) by >=0.05.
G5 transfer: rotated-context retention >=0.90 and aligned-s CI lower >0.90.
Unit=seed with two mirrored history arms; bootstrap resamples seed clusters, seed=20260824, B=10000.
"""
def args():p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',default='results_c');return p.parse_args()
def cluster_ci(rows,key):
 by={s:np.mean([r[key] for r in rows if r['seed']==s]) for s in SEEDS};x=np.array(list(by.values()));rng=np.random.default_rng(BOOT_SEED);m=x[rng.integers(0,len(x),(BOOT_N,len(x)))].mean(1)
 return {'mean':float(x.mean()),'low':float(np.quantile(m,.025)),'high':float(np.quantile(m,.975))}
def auc(score,y):
 score=np.asarray(score);y=np.asarray(y);pos=score[y==1];neg=score[y==0]
 if not len(pos) or not len(neg):return float('nan')
 return float(np.mean([(a>neg).mean()+.5*(a==neg).mean() for a in pos]))
def write(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run_seed(seed,cfg):
 train=data(seed,cfg.train_n,.61);test=data(seed+100000,cfg.test_n,.61);tt=data(seed+200000,cfg.train_n,-.37);te=data(seed+300000,cfg.test_n,-.37)
 rows=[];summ=[]
 for target,sign in [('d',1),('h',-1)]:
  m=LinearSharedModel(np.random.default_rng(seed+400000));k=v=float(sign)
  k,v,*_=step_loop(m,train,test,HISTORY,seed+1,cfg,k,v,target);k,v,dd,dh,s,p=step_loop(m,train,test,COMMON,seed+2,cfg,k,v,None);ps=1 if s>0 else -1
  summ.append({'seed':seed,'history':target,'measure':'base','K':-1,'pre_s':s,'end_s':s,'aligned_end':abs(s),'switched':0,'relax_time':0})
  ms=copy.deepcopy(m);ks,vs=k,v;ks,vs,dds,dhs,ss,pps=step_loop(ms,train,test,SPONTANEOUS,seed+3,cfg,ks,vs,None)
  summ.append({'seed':seed,'history':target,'measure':'spontaneous','K':-1,'pre_s':s,'end_s':ss,'aligned_end':ss*ps,'switched':int(ss*ps<0),'relax_time':0})
  mt=copy.deepcopy(m);kt,vt=k,v;kt,vt,ddt,dht,st,ppt=step_loop(mt,tt,te,COMMON,seed+4,cfg,kt,vt,None)
  summ.append({'seed':seed,'history':target,'measure':'transfer','K':-1,'pre_s':s,'end_s':st,'aligned_end':st*ps,'switched':int(st*ps<0),'relax_time':0})
  for K in KS:
   mp=copy.deepcopy(m);kp,vp=k,v;opp='h' if ps>0 else 'd'
   if K:kp,vp,dd0,dh0,s0,p0=step_loop(mp,train,test,K,seed+100+K,cfg,kp,vp,opp)
   else:dd0,dh0,s0,p0=dd,dh,s,p
   rows.append({'seed':seed,'history':target,'K':K,'pre_sign':ps,'aligned_s':s0*ps,'D_d':dd0,'D_h':dh0,'aligned_kappa':kp*ps,'value':vp,'p_d':p0})
   relax=0
   if K==1600:
    sr=s0
    for elapsed in range(100,RECOVERY+1,100):
     kp,vp,ddr,dhr,sr,pr=step_loop(mp,train,test,100,seed+9000+elapsed,cfg,kp,vp,None)
     if relax==0 and sr*ps>.8:relax=elapsed
   else:kp,vp,ddr,dhr,sr,pr=step_loop(mp,train,test,RECOVERY,seed+500+K,cfg,kp,vp,None)
   summ.append({'seed':seed,'history':target,'measure':'perturbation','K':K,'pre_s':s,'end_s':sr,'aligned_end':sr*ps,'switched':int(sr*ps<0),'relax_time':relax if relax else RECOVERY+1})
 return rows,summ
def main():
 a=args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);cfg=Settings();rows=[];summ=[]
 for seed in SEEDS:
  r,s=run_seed(seed,cfg);rows+=r;summ+=s
 write(out/'perturbation_endpoints.csv',rows);write(out/'summary.csv',summ)
 base=[r for r in summ if r['measure']=='base'];sp=[r for r in summ if r['measure']=='spontaneous'];tr=[r for r in summ if r['measure']=='transfer'];pert=[r for r in summ if r['measure']=='perturbation']
 base_ci=cluster_ci(base,'aligned_end');transfer_ci=cluster_ci(tr,'aligned_end');correct=sum((r['history']=='d' and r['pre_s']>0) or (r['history']=='h' and r['pre_s']<0) for r in base)
 rates={K:np.mean([r['switched'] for r in pert if r['K']==K]) for K in KS};ordered=[rates[k] for k in (3200,4000,4800,5600,6400)]
 relax=[r['relax_time'] for r in pert if r['K']==1600 and not r['switched']];q90=float(np.quantile(relax,.90));return_rate=float(np.mean([not r['switched'] for r in pert if r['K']==1600]))
 mid=[r for r in rows if r['K'] in (4000,4800,5600)];labels=[]
 for r in mid:
  y=next(x['switched'] for x in pert if x['seed']==r['seed'] and x['history']==r['history'] and x['K']==r['K']);labels.append(y)
 auc_q=auc([-r['aligned_kappa'] for r in mid],labels);auc_s=auc([-r['aligned_s'] for r in mid],labels)
 smeans=[np.mean([r['aligned_s'] for r in rows if r['K']==K]) for K in (4000,4800,5600)];srange=max(smeans)-min(smeans);rrange=max(rates[k] for k in (4000,4800,5600))-min(rates[k] for k in (4000,4800,5600))
 gates={'G1_two_states':{'pass':correct==80 and base_ci['low']>.90,'correct_history_arms':correct,'base_abs_s':base_ci},
        'G2_timescale_separation':{'pass':sum(r['switched'] for r in sp)==0 and return_rate>=.90 and q90<=500 and SPONTANEOUS/max(q90,1)>=16,'spontaneous_switches':sum(r['switched'] for r in sp),'return_rate':return_rate,'relax_q90':q90,'timescale_ratio_lower':SPONTANEOUS/max(q90,1)},
        'G3_finite_transition':{'pass':all(a<=b for a,b in zip(ordered,ordered[1:])) and rates[3200]<=.10 and rates[6400]>=.90 and rates[6400]-rates[3200]>=.80,'rates':rates},
        'G4_order_parameter':{'pass':srange<.01 and rrange>=.40 and auc_q>=.70 and auc_q-auc_s>=.05,'surface_range':srange,'rate_range':rrange,'auc_q':auc_q,'auc_surface':auc_s,'auc_gain':auc_q-auc_s},
        'G5_context_transfer':{'pass':np.mean([not r['switched'] for r in tr])>=.90 and transfer_ci['low']>.90,'retention':float(np.mean([not r['switched'] for r in tr])),'aligned_s':transfer_ci}}
 comp={'endpoint_rows':len(rows),'expected_endpoints':len(SEEDS)*2*len(KS),'summary_rows':len(summ),'expected_summary':len(SEEDS)*2*(3+len(KS))};comp['pass']=comp['endpoint_rows']==comp['expected_endpoints'] and comp['summary_rows']==comp['expected_summary']
 if not comp['pass']:raise RuntimeError(str(comp))
 result={'experiment':'III-C','seeds':list(SEEDS),'preregistered_gates':GATES.strip(),'gates':gates,'completeness':comp}
 json_default=lambda x:x.item() if isinstance(x,np.generic) else str(x)
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=json_default),encoding='utf-8')
 lines=['# 実験III-C 結果：多変数閉ループと秩序変数候補','','## 仮説群','']
 for k,v in gates.items():lines.append(f"- **{k}**: {'PASS' if v['pass'] else 'FAIL'}")
 lines += ['','## 転換率','','|K|転換率|','|---:|---:|']+[f'|{k}|{rates[k]:.3f}|' for k in KS]
 lines += ['','## 主要診断',f'- 自発転換: {gates["G2_timescale_separation"]["spontaneous_switches"]}/80',f'- 復帰時間90%分位: {q90:.0f}',f'- 時間尺度比の下限: {gates["G2_timescale_separation"]["timescale_ratio_lower"]:.1f}',f'- q AUC: {auc_q:.3f} / 表面指標AUC: {auc_s:.3f}',f'- 別表層環境での枝維持率: {gates["G5_context_transfer"]["retention"]:.3f}','','更新則は外生である。qは構成系内の秩序変数候補であり、人間的志向性の測定量ではない。']
 (out/'result_c.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 prov={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'script_sha256':sha256(Path(__file__).resolve()),'preregistration_sha256':sha256(Path(__file__).parent/'preregistration_c.md')}
 (out/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'gates':{k:v['pass'] for k,v in gates.items()},'completeness':comp,'output':str(out)},ensure_ascii=False,indent=2,default=json_default))
if __name__=='__main__':main()
