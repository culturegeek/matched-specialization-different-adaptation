#!/usr/bin/env python3
"""III-A2b 設計パイロット: 表現容量と一更新で学べる対象数を分離する。"""
from __future__ import annotations
import copy,csv
from pathlib import Path
import numpy as np
from pilot_a2_capacity_phases import LinearSharedModel,Settings,dataset,metrics

SEEDS=range(3320,3326)
CELLS={"C1_single":(1,"single"),"C1_both":(1,"both"),"C2_single":(2,"single"),"C2_both":(2,"both")}

def phase(model,train,test,capacity,mode,steps,seed,cfg,stabilized):
 rngb=np.random.default_rng(seed);rngs=np.random.default_rng(seed+10_000_000);p=.5
 for step in range(steps):
  if stabilized and step%cfg.selector_interval==0:
   _,_,s=metrics(model,test,capacity);p=float(np.clip(.5+cfg.selector_bias*np.tanh(cfg.selector_gain*s),.1,.9))
  idx=rngb.integers(0,len(train[0]),size=cfg.batch);b=tuple(v[idx] for v in train)
  if mode=="both":model.step(*b,"d",capacity,cfg);model.step(*b,"h",capacity,cfg)
  else:model.step(*b,("d" if rngs.random()<p else "h"),capacity,cfg)
 return metrics(model,test,capacity)

def main():
 cfg=Settings();rows=[]
 for seed in SEEDS:
  train=dataset(seed,cfg.train_n);test=dataset(seed+100_000,cfg.test_n);initial=LinearSharedModel(np.random.default_rng(seed+200_000))
  for name,(cap,mode) in CELLS.items():
   m=copy.deepcopy(initial);phase(m,train,test,cap,mode,cfg.balanced_steps,seed+1,cfg,False)
   dd,dh,s=phase(m,train,test,cap,mode,cfg.formation_steps,seed+2,cfg,True)
   rows.append({"seed":seed,"cell":name,"capacity":cap,"update_mode":mode,"D_d":dd,"D_h":dh,"sum_D":dd+dh,"s":s,"abs_s":abs(s)})
 path=Path("pilot_a2b_results.csv")
 with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(path)
if __name__=="__main__":main()
