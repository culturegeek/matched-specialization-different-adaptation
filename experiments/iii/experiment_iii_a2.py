#!/usr/bin/env python3
"""実験III-A2: 形成期×測定期の共同資源拘束2×2確認実験。"""

from __future__ import annotations

import argparse, copy, csv, hashlib, json, platform, sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiment_iii_a0 import sha256
from pilot_a2_capacity_phases import LinearSharedModel, Settings, CONDITIONS, dataset, metrics
from pilot_a2_capacity_phases_v2 import train_phase


SEEDS=tuple(range(3400,3440)); BOOT_SEED=20260824; BOOT_N=10000
CHECKPOINTS=(300,900,1800,2400,3600)
PREREGISTERED_GATES="""
G1 formation: binding formation |s| CI lower >0.80; unbinding formation |s| CI upper <0.10.
G2 current regime: terminal CC and UC |s| CI lower >0.80; UU CI upper <0.10.
G3 release: CU |s| reduction CI lower >0.70; CU terminal |s| CI upper <0.20.
G4 symmetry: pooled terminal constrained positive-branch fraction in [0.25,0.75].
Unit=seed 3400--3439; paired percentile bootstrap seed=20260824, B=10000.
"""


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",default="results_a2");return p.parse_args()


def ci(x):
    x=np.asarray(x,float);rng=np.random.default_rng(BOOT_SEED);idx=rng.integers(0,len(x),(BOOT_N,len(x)))
    m=x[idx].mean(1);return {"mean":float(x.mean()),"low":float(np.quantile(m,.025)),"high":float(np.quantile(m,.975))}


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)


def main():
    args=parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);cfg=Settings();rows=[]
    for seed in SEEDS:
        train=dataset(seed,cfg.train_n);test=dataset(seed+100_000,cfg.test_n)
        initial=LinearSharedModel(np.random.default_rng(seed+200_000))
        for name,(form,measure) in CONDITIONS.items():
            model=copy.deepcopy(initial)
            train_phase(model,train,test,form,cfg.balanced_steps,seed+1,cfg,False)
            d1,d2,s_form=train_phase(model,train,test,form,cfg.formation_steps,seed+2,cfg,True)
            rows.append({"seed":seed,"condition":name,"phase":"formation_end","step":0,
                         "regime":form,"D_d":d1,"D_h":d2,"s":s_form,"abs_s":abs(s_form)})
            done=0
            for cp in CHECKPOINTS:
                d1,d2,s=train_phase(model,train,test,measure,cp-done,seed+3+done,cfg,True);done=cp
                rows.append({"seed":seed,"condition":name,"phase":"measurement","step":cp,
                             "regime":measure,"D_d":d1,"D_h":d2,"s":s,"abs_s":abs(s)})
    write_csv(out/"trajectories.csv",rows)
    form_rows=[r for r in rows if r["phase"]=="formation_end"]
    term={c:[r for r in rows if r["condition"]==c and r["phase"]=="measurement" and r["step"]==3600]
          for c in CONDITIONS}
    # Conditions sharing the same formation regime are identical through formation; use one row per seed.
    form_c=[r["abs_s"] for r in form_rows if r["condition"]=="CC"]
    form_u=[r["abs_s"] for r in form_rows if r["condition"]=="UU"]
    stats={"formation_C":ci(form_c),"formation_U":ci(form_u)}
    for c in CONDITIONS:stats[f"terminal_{c}"]=ci([r["abs_s"] for r in term[c]])
    cu_form={r["seed"]:r["abs_s"] for r in form_rows if r["condition"]=="CU"}
    cu_reduction=[cu_form[r["seed"]]-r["abs_s"] for r in term["CU"]]
    stats["CU_reduction"]=ci(cu_reduction)
    constrained=term["CC"]+term["UC"]
    pos_frac=float(np.mean([r["s"]>0 for r in constrained]))
    gates={
      "G1_formation_resource": {"pass":stats["formation_C"]["low"]>0.80 and stats["formation_U"]["high"]<0.10},
      "G2_current_resource": {"pass":stats["terminal_CC"]["low"]>0.80 and stats["terminal_UC"]["low"]>0.80 and stats["terminal_UU"]["high"]<0.10},
      "G3_release_history": {"pass":stats["CU_reduction"]["low"]>0.70 and stats["terminal_CU"]["high"]<0.20},
      "G4_symmetry": {"pass":0.25<=pos_frac<=0.75,"positive_branch_fraction":pos_frac},
    }
    complete={"rows":len(rows),"expected":len(SEEDS)*len(CONDITIONS)*(1+len(CHECKPOINTS)),
              "unique_keys":len({(r['seed'],r['condition'],r['phase'],r['step']) for r in rows})}
    complete["pass"]=complete["rows"]==complete["expected"]==complete["unique_keys"]
    if not complete["pass"]:raise RuntimeError(f"完全性検査に失敗:{complete}")
    result={"experiment":"III-A2","seeds":list(SEEDS),"settings":asdict(cfg),
            "preregistered_gates":PREREGISTERED_GATES.strip(),"statistics":stats,"gates":gates,"completeness":complete}
    (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 実験III-A2 結果：形成期×測定期の資源拘束","","## 仮説群",""]
    for k,v in gates.items():lines.append(f"- **{k}**: {'PASS' if v['pass'] else 'FAIL'}")
    lines += ["","## |s| の区間","","|測定量|平均|95%区間|","|---|---:|---:|"]
    for k,v in stats.items():lines.append(f"|{k}|{v['mean']:.6f}|[{v['low']:.6f}, {v['high']:.6f}]|")
    lines += ["",f"拘束測定条件の正枝割合: {pos_frac:.3f}","","## 進行判断",""]
    failed=[k for k,v in gates.items() if not v["pass"]]
    lines.append("全仮説群が成立した。A2bで表現容量と学習機会を分離する。" if not failed else
                 "不成立要因を監査する: "+", ".join(failed))
    lines += ["","この結果は共同資源条件の効果であり、表現容量と学習機会の個別寄与はまだ分離していない。"]
    (out/"result_a2.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    prov={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
          "script_sha256":sha256(Path(__file__).resolve()),
          "preregistration_sha256":sha256(Path(__file__).parent/"preregistration_a2.md")}
    (out/"provenance.json").write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"gates":{k:v["pass"] for k,v in gates.items()},"completeness":complete,"output":str(out)},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
