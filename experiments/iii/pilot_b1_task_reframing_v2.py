#!/usr/bin/env python3
"""III-B1 設計パイロットv2: 潜在設備状態を用いた課題再編の4要因分離。"""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from experiment_iii_a0 import Config,STATES,initial_distribution,transition_and_cost

SEEDS=range(3610,3616);STEPS=400
FACTORS=list(itertools.product((0,1),(0,1),(1,10),(0.10,1.0)))
IDX={s:i for i,s in enumerate(STATES)}

def posterior(prior,signal,accuracy):
 like1=accuracy if signal==1 else 1-accuracy;like0=(1-accuracy) if signal==1 else accuracy
 return prior*like1/(prior*like1+(1-prior)*like0+1e-12)
def next_belief(prior,action,cfg):
 requested,maintain=action
 if maintain:return prior*(1-cfg.repair_success)+(1-prior)*.02
 p1=cfg.health_persist_no_use+.02*requested;p0=cfg.wear_base+cfg.wear_per_delivery*requested
 return prior*min(.99,p1)+(1-prior)*min(.95,p0)
def draw(prob,u):return int(np.searchsorted(np.cumsum(prob),u,side='right'))

def run(seed,explore,plastic,hpred,heval):
 cfg=Config();actions,p,c=transition_and_cost(cfg,False);rng=np.random.default_rng(seed)
 state=draw(initial_distribution(cfg),rng.random());belief=cfg.initial_worn;frame=0;switches=0
 tot=np.zeros(2);health_frames=[];worn=[];correct=[]
 acc=cfg.sensor_accuracy if explore else cfg.standard_accuracy
 horizon_mass=sum(heval**t for t in range(hpred))
 for _ in range(STEPS):
  stock,d,h=STATES[state];signal=h if rng.random()<acc else 1-h;belief=posterior(belief,signal,acc)
  evidence=belief*horizon_mass
  if plastic:
   if frame==0 and evidence>3.0:frame=1;switches+=1
   elif frame==1 and evidence<1.4:frame=0;switches+=1
  # Demand frame cannot use health belief in action formation; health frame can.
  if frame==0:action_i=2 if stock>=2 else (1 if stock>=1 else 0)
  else:action_i=4 if belief>.45 else 1
  tot+=c[action_i,state,:2];health_frames.append(frame);worn.append(h);correct.append(int(frame==h))
  belief=next_belief(belief,actions[action_i],cfg);state=draw(p[action_i,state],rng.random())
 return {'switch_rate':switches/STEPS,'health_frame_fraction':float(np.mean(health_frames)),
         'frame_health_alignment':float(np.mean(correct)),'shortage':tot[0]/STEPS,'risk':tot[1]/STEPS,
         'balanced_objective':tot.sum()/(2*STEPS),'worn_rate':float(np.mean(worn))}

def main():
 rows=[]
 for seed in SEEDS:
  for e,pk,hp,he in FACTORS:rows.append({'seed':seed,'explore':e,'plastic':pk,'Hpred':hp,'Heval':he,**run(seed,e,pk,hp,he)})
 path=Path('pilot_b1_v2_results.csv')
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(path)
if __name__=='__main__':main()
