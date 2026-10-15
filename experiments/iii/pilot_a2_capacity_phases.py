#!/usr/bin/env python3
"""III-A2 設計パイロット: 形成期×測定期の表現容量2×2。

探索用 seed 3300--3305。判定条件は置かず、確認実験の時間幅と測定量を決める。
"""

from __future__ import annotations

import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SEEDS = range(3300, 3306)
CONDITIONS = {"UU": (2, 2), "UC": (2, 1), "CU": (1, 2), "CC": (1, 1)}


@dataclass
class Settings:
    train_n: int = 2048
    test_n: int = 2048
    batch: int = 128
    balanced_steps: int = 300
    formation_steps: int = 1800
    measurement_steps: int = 3600
    lr: float = 0.018
    selector_interval: int = 10
    selector_gain: float = 3.0
    selector_bias: float = 0.40
    weight_decay: float = 1e-4


class LinearSharedModel:
    def __init__(self, rng):
        self.encoder = rng.normal(0, 0.20, size=(2, 2))
        self.head_d = rng.normal(0, 0.20, size=2)
        self.head_h = rng.normal(0, 0.20, size=2)

    def predict(self, x, capacity):
        mask = np.array([1.0, 1.0 if capacity == 2 else 0.0])
        pre = x @ self.encoder
        z = np.tanh(pre) * mask
        return pre, z, z @ self.head_d, z @ self.head_h, mask

    def step(self, x, y_d, y_h, target, capacity, cfg):
        pre, z, pd, ph, mask = self.predict(x, capacity)
        y = y_d if target == "d" else y_h
        head = self.head_d if target == "d" else self.head_h
        pred = pd if target == "d" else ph
        grad_pred = 2.0 * (pred - y) / len(x)
        grad_head = z.T @ grad_pred + cfg.weight_decay * head
        grad_z = grad_pred[:, None] * head[None, :]
        grad_pre = grad_z * (1 - np.tanh(pre) ** 2) * mask
        grad_encoder = x.T @ grad_pre + cfg.weight_decay * self.encoder
        self.encoder -= cfg.lr * grad_encoder
        if target == "d": self.head_d -= cfg.lr * grad_head
        else: self.head_h -= cfg.lr * grad_head


def dataset(seed, n):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 2))
    angle = 0.61
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    x = latent @ rot + 0.03 * rng.normal(size=(n, 2))
    return x, latent[:, 0], latent[:, 1]


def metrics(model, data, capacity):
    x, d, h = data
    _, _, pd, ph, _ = model.predict(x, capacity)
    dd = float(np.mean((pd-d)**2) / np.var(d))
    dh = float(np.mean((ph-h)**2) / np.var(h))
    s = (dh-dd)/(dh+dd+1e-12)
    return dd, dh, s


def train_phase(model, train, test, capacity, steps, seed, cfg, stabilized):
    rng_b = np.random.default_rng(seed)
    rng_s = np.random.default_rng(seed + 10_000_000)
    p_d = 0.5
    for step in range(steps):
        if stabilized and step % cfg.selector_interval == 0:
            _, _, s = metrics(model, test, capacity)
            p_d = float(np.clip(0.5 + cfg.selector_bias * np.tanh(cfg.selector_gain*s), 0.1, 0.9))
        elif not stabilized:
            p_d = 0.5
        idx = rng_b.integers(0, len(train[0]), size=cfg.batch)
        target = "d" if rng_s.random() < p_d else "h"
        model.step(*(v[idx] for v in train), target, capacity, cfg)
    return metrics(model, test, capacity)


def main():
    cfg = Settings(); rows=[]
    checkpoints=(0, 300, 900, 1800, 2400, 3600)
    for seed in SEEDS:
        train=dataset(seed, cfg.train_n); test=dataset(seed+100_000, cfg.test_n)
        init_rng=np.random.default_rng(seed+200_000)
        initial=LinearSharedModel(init_rng)
        for name,(c_form,c_measure) in CONDITIONS.items():
            model=copy.deepcopy(initial)
            train_phase(model,train,test,c_form,cfg.balanced_steps,seed+1,cfg,False)
            d1,d2,s_form=train_phase(model,train,test,c_form,cfg.formation_steps,seed+2,cfg,True)
            done=0
            rows.append({"seed":seed,"condition":name,"phase":"formation_end","step":0,
                         "capacity":c_form,"D_d":d1,"D_h":d2,"s":s_form,"abs_s":abs(s_form)})
            for cp in checkpoints[1:]:
                d1,d2,s=train_phase(model,train,test,c_measure,cp-done,seed+3+done,cfg,True); done=cp
                rows.append({"seed":seed,"condition":name,"phase":"measurement","step":cp,
                             "capacity":c_measure,"D_d":d1,"D_h":d2,"s":s,"abs_s":abs(s)})
    path=Path("pilot_a2_results.csv")
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    print(path)


if __name__=="__main__": main()
