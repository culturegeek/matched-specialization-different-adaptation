#!/usr/bin/env python3
"""III-B1 設計パイロット: 探索×課題可塑性×予測地平×評価地平。"""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from experiment_iii_a0 import Config,STATES,initial_distribution,transition_and_cost

SEEDS=range(3600,3606);STEPS=240
FACTORS=list(itertools.product((0,1),(0,1),(1,10),(0.10,1.0)))

def first_policy(p,c,horizon,discount,weights,allowed):
 stage=weights[0]*c[:,:,0]+weights[1]*c[:,:,1];v=np.zeros(len(STATES));act=None
 for _ in range(horizon):
  q=stage+discount*np.einsum('ask,k->as',p,v);qa=q[allowed];loc=np.argmin(qa,axis=0);act=np.array(allowed)[loc];v=q[act,np.arange(len(STATES))]
 return act

def predicted_components(p,c,policy,state,horizon,discount):
 occ=np.zeros(len(STATES));occ[state]=1.;tot=np.zeros(2);scale=0.
 for t in range(horizon):
  a=policy;w=discount**t;tot+=w*np.sum(occ[:,None]*c[a,np.arange(len(STATES)),:2],axis=0);scale+=w
  nxt=np.zeros_like(occ)
  for s in range(len(STATES)):nxt+=occ[s]*p[a[s],s]
  occ=nxt
 return tot/max(scale,1e-12)

def draw_next(prob,u):return int(np.searchsorted(np.cumsum(prob),u,side='right'))

def run(seed,explore,plastic,hpred,gamma):
 cfg=Config();actions,p,c=transition_and_cost(cfg,False);rng=np.random.default_rng(seed)
 state=draw_next(initial_distribution(cfg),rng.random());frame=0;switches=0;tot=np.zeros(2);frames=[];acts=[]
 allowed=list(range(len(actions))) if explore else [0,1,2]
 for t in range(STEPS):
  weights=(.82,.18) if frame==0 else (.18,.82)
  pol=first_policy(p,c,hpred,gamma,weights,allowed)
  pred=predicted_components(p,c,pol,state,hpred,gamma)
  # Normalize by A0 outcome ranges. Reframe if the unattended problem dominates.
  scores=np.array([pred[0]/0.40,pred[1]/0.155])
  other=1-frame
  if plastic and scores[other]>scores[frame]+0.12:
   frame=other;switches+=1;weights=(.82,.18) if frame==0 else (.18,.82)
   pol=first_policy(p,c,hpred,gamma,weights,allowed)
  action=int(pol[state]);tot+=c[action,state,:2];acts.append(action);frames.append(frame)
  state=draw_next(p[action,state],rng.random())
 return {'switch_rate':switches/STEPS,'switches':switches,'shortage':tot[0]/STEPS,'risk':tot[1]/STEPS,
         'balanced_objective':tot.sum()/(2*STEPS),'health_frame_fraction':np.mean(frames),
         'action_diversity':len(set(acts))}

def main():
 rows=[]
 for seed in SEEDS:
  for e,pk,hp,he in FACTORS:rows.append({'seed':seed,'explore':e,'plastic':pk,'Hpred':hp,'Heval':he,**run(seed,e,pk,hp,he)})
 path=Path('pilot_b1_results.csv')
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(path)
if __name__=='__main__':main()
