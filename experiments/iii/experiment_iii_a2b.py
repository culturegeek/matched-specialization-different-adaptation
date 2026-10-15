#!/usr/bin/env python3
"""実験III-A2b: 表現容量と学習機会の2×2確認実験。"""
from __future__ import annotations
import argparse,copy,csv,json,platform,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from experiment_iii_a0 import sha256
from pilot_a2_capacity_phases import LinearSharedModel,Settings,dataset
from pilot_a2b_resource_components import CELLS,phase

SEEDS=tuple(range(3500,3540));BOOT_SEED=20260824;BOOT_N=10000
GATES="""
G1 representation: C1_both |s| CI lower >0.80 and sum_D gap C1_both-C2_both CI lower >0.80.
G2 learning opportunity: |s| paired difference C2_single-C2_both CI lower >0.15.
G3 nonbinding: C2_both |s| CI upper <0.20 and sum_D CI upper <0.05.
G4 symmetry: C1_both positive-branch fraction in [0.25,0.75].
Unit=seed 3500--3539; percentile bootstrap seed=20260824, B=10000.
"""
def args():p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',default='results_a2b');return p.parse_args()
def ci(x):
 x=np.asarray(x,float);rng=np.random.default_rng(BOOT_SEED);m=x[rng.integers(0,len(x),(BOOT_N,len(x)))].mean(1)
 return {'mean':float(x.mean()),'low':float(np.quantile(m,.025)),'high':float(np.quantile(m,.975))}
def main():
 a=args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);cfg=Settings();rows=[]
 for seed in SEEDS:
  train=dataset(seed,cfg.train_n);test=dataset(seed+100_000,cfg.test_n);initial=LinearSharedModel(np.random.default_rng(seed+200_000))
  for name,(cap,mode) in CELLS.items():
   m=copy.deepcopy(initial);phase(m,train,test,cap,mode,cfg.balanced_steps,seed+1,cfg,False)
   dd,dh,s=phase(m,train,test,cap,mode,cfg.formation_steps,seed+2,cfg,True)
   rows.append({'seed':seed,'cell':name,'capacity':cap,'update_mode':mode,'D_d':dd,'D_h':dh,'sum_D':dd+dh,'s':s,'abs_s':abs(s)})
 with (out/'cells.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 by={c:{r['seed']:r for r in rows if r['cell']==c} for c in CELLS}
 stats={}
 for c in CELLS:stats[c+'_abs_s']=ci([by[c][s]['abs_s'] for s in SEEDS]);stats[c+'_sum_D']=ci([by[c][s]['sum_D'] for s in SEEDS])
 stats['representation_sumD_gap']=ci([by['C1_both'][s]['sum_D']-by['C2_both'][s]['sum_D'] for s in SEEDS])
 stats['learning_absS_gap']=ci([by['C2_single'][s]['abs_s']-by['C2_both'][s]['abs_s'] for s in SEEDS])
 stats['C1_single_minus_both']=ci([by['C1_single'][s]['abs_s']-by['C1_both'][s]['abs_s'] for s in SEEDS])
 interaction=[(by['C1_single'][s]['abs_s']-by['C1_both'][s]['abs_s'])-(by['C2_single'][s]['abs_s']-by['C2_both'][s]['abs_s']) for s in SEEDS]
 stats['interaction']=ci(interaction);pos=float(np.mean([by['C1_both'][s]['s']>0 for s in SEEDS]))
 gates={
  'G1_representation_capacity':{'pass':stats['C1_both_abs_s']['low']>0.80 and stats['representation_sumD_gap']['low']>0.80},
  'G2_learning_opportunity':{'pass':stats['learning_absS_gap']['low']>0.15},
  'G3_nonbinding_control':{'pass':stats['C2_both_abs_s']['high']<0.20 and stats['C2_both_sum_D']['high']<0.05},
  'G4_symmetry':{'pass':0.25<=pos<=0.75,'positive_fraction':pos}}
 comp={'rows':len(rows),'expected':len(SEEDS)*len(CELLS),'unique':len({(r['seed'],r['cell']) for r in rows})};comp['pass']=comp['rows']==comp['expected']==comp['unique']
 if not comp['pass']:raise RuntimeError(str(comp))
 result={'experiment':'III-A2b','seeds':list(SEEDS),'settings':asdict(cfg),'preregistered_gates':GATES.strip(),'statistics':stats,'gates':gates,'completeness':comp}
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# 実験III-A2b 結果：表現容量と学習機会の分離','','## 仮説群','']
 for k,v in gates.items():lines.append(f"- **{k}**: {'PASS' if v['pass'] else 'FAIL'}")
 lines += ['','## 測定量','','|測定量|平均|95%区間|','|---|---:|---:|']
 for k,v in stats.items():lines.append(f"|{k}|{v['mean']:.6f}|[{v['low']:.6f}, {v['high']:.6f}]|")
 lines += ['',f'C1_bothの正枝割合: {pos:.3f}','','この結果は線形共有表現を用いた構成系での分離であり、一般的必要性の証明ではない。']
 (out/'result_a2b.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 prov={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'script_sha256':sha256(Path(__file__).resolve()),'preregistration_sha256':sha256(Path(__file__).parent/'preregistration_a2b.md')}
 (out/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'gates':{k:v['pass'] for k,v in gates.items()},'completeness':comp,'output':str(out)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
