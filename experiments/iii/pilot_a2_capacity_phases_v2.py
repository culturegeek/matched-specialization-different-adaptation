#!/usr/bin/env python3
"""III-A2 設計パイロットv2: 拘束・非拘束を容量と学習機会の組で定義する。

探索用 seed 3310--3315。非拘束は容量2かつ両対象更新、拘束は容量1かつ一対象更新。
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np

from pilot_a2_capacity_phases import LinearSharedModel, Settings, CONDITIONS, dataset, metrics


SEEDS=range(3310,3316)


def train_phase(model,train,test,regime,steps,seed,cfg,stabilized):
    capacity = 2 if regime == 2 else 1
    rng_b=np.random.default_rng(seed); rng_s=np.random.default_rng(seed+10_000_000); p_d=.5
    for step in range(steps):
        if stabilized and step%cfg.selector_interval==0:
            _,_,s=metrics(model,test,capacity)
            p_d=float(np.clip(.5+cfg.selector_bias*np.tanh(cfg.selector_gain*s),.1,.9))
        idx=rng_b.integers(0,len(train[0]),size=cfg.batch)
        batch=tuple(v[idx] for v in train)
        if regime==2:
            # 非拘束: 同じ経験から両推定対象を更新できる。
            model.step(*batch,"d",capacity,cfg)
            model.step(*batch,"h",capacity,cfg)
        else:
            model.step(*batch,("d" if rng_s.random()<p_d else "h"),capacity,cfg)
    return metrics(model,test,capacity)


def main():
    cfg=Settings();rows=[];checkpoints=(0,300,900,1800,2400,3600)
    for seed in SEEDS:
        train=dataset(seed,cfg.train_n);test=dataset(seed+100_000,cfg.test_n)
        initial=LinearSharedModel(np.random.default_rng(seed+200_000))
        for name,(form,measure) in CONDITIONS.items():
            model=copy.deepcopy(initial)
            train_phase(model,train,test,form,cfg.balanced_steps,seed+1,cfg,False)
            d1,d2,s=train_phase(model,train,test,form,cfg.formation_steps,seed+2,cfg,True)
            rows.append({"seed":seed,"condition":name,"phase":"formation_end","step":0,
                         "regime":form,"D_d":d1,"D_h":d2,"s":s,"abs_s":abs(s)})
            done=0
            for cp in checkpoints[1:]:
                d1,d2,s=train_phase(model,train,test,measure,cp-done,seed+3+done,cfg,True);done=cp
                rows.append({"seed":seed,"condition":name,"phase":"measurement","step":cp,
                             "regime":measure,"D_d":d1,"D_h":d2,"s":s,"abs_s":abs(s)})
    path=Path("pilot_a2_v2_results.csv")
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    print(path)


if __name__=="__main__":main()
