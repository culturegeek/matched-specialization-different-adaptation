#!/usr/bin/env python3
"""III-C 設計パイロット: 課題枠組みを遅い変数として含む多変数閉ループ。"""
from __future__ import annotations
import copy,csv,math
from pathlib import Path
import numpy as np
from pilot_a2_capacity_phases import LinearSharedModel,Settings,metrics

SEEDS=range(4000,4006);HISTORY=1200;COMMON=4000;RECOVERY=4000;SPONTANEOUS=8000
KS=(0,100,400,800,1600,3200,4000,4800,5600,6400,12800);CAPACITY=1

def data(seed,n,angle):
 rng=np.random.default_rng(seed);lat=rng.normal(size=(n,2));rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
 return lat@rot+.03*rng.normal(size=(n,2)),lat[:,0],lat[:,1]
def update_slow(kappa,value,s):
 kappa=.998*kappa+.002*np.tanh(3*s);value=.999*value+.001*kappa
 return float(kappa),float(value)
def p_select(s,kappa,value,cfg):return float(np.clip(.5+cfg.selector_bias*np.tanh(2*s+1.2*kappa+.5*value),.1,.9))
def step_loop(model,train,test,n,seed,cfg,kappa,value,forced=None):
 rb=np.random.default_rng(seed);rs=np.random.default_rng(seed+10_000_000);p=.5
 for t in range(n):
  if t%cfg.selector_interval==0:
   _,_,s=metrics(model,test,CAPACITY);kappa,value=update_slow(kappa,value,s);p=p_select(s,kappa,value,cfg)
  idx=rb.integers(0,len(train[0]),size=cfg.batch);target=forced if forced else ('d' if rs.random()<p else 'h')
  model.step(*(v[idx] for v in train),target,CAPACITY,cfg)
 dd,dh,s=metrics(model,test,CAPACITY);return kappa,value,dd,dh,s,p
def main():
 cfg=Settings();rows=[];summary=[]
 for seed in SEEDS:
  train=data(seed,cfg.train_n,.61);test=data(seed+100000,cfg.test_n,.61);transfer_train=data(seed+200000,cfg.train_n,-.37);transfer_test=data(seed+300000,cfg.test_n,-.37)
  for history_target,sign in [('d',1),('h',-1)]:
   m=LinearSharedModel(np.random.default_rng(seed+400000));kappa=float(sign);value=float(sign)
   kappa,value,*_=step_loop(m,train,test,HISTORY,seed+1,cfg,kappa,value,history_target)
   kappa,value,dd,dh,s,p=step_loop(m,train,test,COMMON,seed+2,cfg,kappa,value,None);pre_sign=1 if s>0 else -1
   base=(copy.deepcopy(m),kappa,value,dd,dh,s,p)
   # Spontaneous residence observation.
   ms=copy.deepcopy(m);ks,vs=kappa,value
   ks,vs,dds,dhs,ss,ps=step_loop(ms,train,test,SPONTANEOUS,seed+3,cfg,ks,vs,None)
   summary.append({'seed':seed,'history':history_target,'measure':'spontaneous','K':-1,'pre_s':s,'end_s':ss,'switched':int(ss*pre_sign<0),'kappa_end':ks})
   # Surface transfer with the same rules and fixed state variables.
   mt=copy.deepcopy(m);kt,vt=kappa,value
   kt,vt,ddt,dht,st,pt=step_loop(mt,transfer_train,transfer_test,COMMON,seed+4,cfg,kt,vt,None)
   summary.append({'seed':seed,'history':history_target,'measure':'transfer','K':-1,'pre_s':s,'end_s':st,'switched':int(st*pre_sign<0),'kappa_end':kt})
   for K in KS:
    mp=copy.deepcopy(base[0]);kp,vp=base[1],base[2];opposite='h' if pre_sign>0 else 'd'
    if K:kp,vp,dd0,dh0,s0,p0=step_loop(mp,train,test,K,seed+100+K,cfg,kp,vp,opposite)
    else:dd0,dh0,s0,p0=dd,dh,s,p
    rows.append({'seed':seed,'history':history_target,'K':K,'stage':'perturb_end','pre_sign':pre_sign,'s':s0,'aligned_s':s0*pre_sign,'D_d':dd0,'D_h':dh0,'kappa':kp,'aligned_kappa':kp*pre_sign,'value':vp,'p_d':p0})
    kr,vr,ddr,dhr,sr,pr=step_loop(mp,train,test,RECOVERY,seed+500+K,cfg,kp,vp,None)
    switched=int(sr*pre_sign<0)
    rows.append({'seed':seed,'history':history_target,'K':K,'stage':'recovery_end','pre_sign':pre_sign,'s':sr,'aligned_s':sr*pre_sign,'D_d':ddr,'D_h':dhr,'kappa':kr,'aligned_kappa':kr*pre_sign,'value':vr,'p_d':pr})
    summary.append({'seed':seed,'history':history_target,'measure':'perturbation','K':K,'pre_s':s,'end_s':sr,'switched':switched,'kappa_end':kr})
 for path,data_rows in ((Path('pilot_c_states.csv'),rows),(Path('pilot_c_summary.csv'),summary)):
  with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(data_rows[0]));w.writeheader();w.writerows(data_rows)
 print('pilot_c_states.csv')
if __name__=='__main__':main()
