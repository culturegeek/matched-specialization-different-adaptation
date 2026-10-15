#!/usr/bin/env python3
"""III-B2 設計パイロット: 最大計算を揃えた停止規則比較。"""
from __future__ import annotations
import csv,math
from pathlib import Path
import numpy as np

SEEDS=range(3800,3806);EPISODES=2000;MAX_STEPS=30;RULES=('fixed','precision','information_gain')
def entropy(p):
 if p<=0 or p>=1:return 0.
 return -p*math.log2(p)-(1-p)*math.log2(1-p)
def update(p,y,acc):
 l1=acc if y else 1-acc;l0=(1-acc) if y else acc
 return p*l1/(p*l1+(1-p)*l0+1e-15)
def expected_ig(p,acc):
 py=p*acc+(1-p)*(1-acc);p1=update(p,1,acc);p0=update(p,0,acc)
 return entropy(p)-(py*entropy(p1)+(1-py)*entropy(p0))
def run(seed,rule):
 rng=np.random.default_rng(seed);rows=[]
 for ep in range(EPISODES):
  truth=int(rng.random()<.5);hard=int(rng.random()<.5);acc=.62 if hard else .88;p=.5;n=0
  for t in range(1,MAX_STEPS+1):
   stop=(rule=='fixed' and n>=12) or (rule=='precision' and n>=2 and entropy(p)<.25) or (rule=='information_gain' and n>=2 and expected_ig(p,acc)<.03)
   if stop:break
   y=truth if rng.random()<acc else 1-truth;p=update(p,y,acc);n+=1
  rows.append((n,int((p>=.5)==truth),(p-truth)**2,hard,MAX_STEPS))
 a=np.array(rows,float)
 return {'n_obs':a[:,0].mean(),'accuracy':a[:,1].mean(),'brier':a[:,2].mean(),
         'n_easy':a[a[:,3]==0,0].mean(),'n_hard':a[a[:,3]==1,0].mean(),'total_compute':a[:,4].mean()}
def main():
 rows=[]
 for s in SEEDS:
  for r in RULES:rows.append({'seed':s,'rule':r,**run(s,r)})
 path=Path('pilot_b2_results.csv')
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(path)
if __name__=='__main__':main()
