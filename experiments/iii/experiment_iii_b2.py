#!/usr/bin/env python3
"""実験III-B2: 総計算量を揃えた停止規則の確認実験。"""
from __future__ import annotations
import argparse,csv,json,platform,sys
from pathlib import Path
import numpy as np
from experiment_iii_a0 import sha256
from pilot_b2_stopping_rules import run,RULES,MAX_STEPS
SEEDS=tuple(range(3900,3940));BOOT_SEED=20260824;BOOT_N=10000
GATES="""
G1 compute matched: total_compute exactly 30 in every seed/rule.
G2 adaptation: precision hard-easy observation gap CI lower >15; fixed absolute gap <0.1.
G3 precision: mean observations within +/-1 of fixed; accuracy gain CI lower >0.03;
   Brier reduction CI lower >0.015.
G4 information gain: observation reduction CI lower >4; accuracy difference CI lower >-0.02.
Unit=seed 3900--3939; percentile bootstrap seed=20260824, B=10000.
"""
def args():p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',default='results_b2');return p.parse_args()
def ci(x):
 x=np.asarray(x,float);rng=np.random.default_rng(BOOT_SEED);m=x[rng.integers(0,len(x),(BOOT_N,len(x)))].mean(1)
 return {'mean':float(x.mean()),'low':float(np.quantile(m,.025)),'high':float(np.quantile(m,.975))}
def main():
 a=args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
 for s in SEEDS:
  for r in RULES:rows.append({'seed':s,'rule':r,**run(s,r)})
 with (out/'rules.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 by={(r['seed'],r['rule']):r for r in rows}
 diff=lambda rule1,rule2,key:[by[(s,rule1)][key]-by[(s,rule2)][key] for s in SEEDS]
 stats={'precision_adaptation':ci([by[(s,'precision')]['n_hard']-by[(s,'precision')]['n_easy'] for s in SEEDS]),
        'fixed_adaptation':ci([by[(s,'fixed')]['n_hard']-by[(s,'fixed')]['n_easy'] for s in SEEDS]),
        'precision_obs_change':ci(diff('precision','fixed','n_obs')),
        'precision_accuracy_gain':ci(diff('precision','fixed','accuracy')),
        'precision_brier_reduction':ci(diff('fixed','precision','brier')),
        'ig_observation_reduction':ci(diff('fixed','information_gain','n_obs')),
        'ig_accuracy_change':ci(diff('information_gain','fixed','accuracy'))}
 matched=all(abs(r['total_compute']-MAX_STEPS)<1e-12 for r in rows)
 gates={'G1_compute_matched':{'pass':matched},
        'G2_adaptive_stopping':{'pass':stats['precision_adaptation']['low']>15 and abs(stats['fixed_adaptation']['mean'])<.1},
        'G3_precision_benefit':{'pass':abs(stats['precision_obs_change']['mean'])<1 and stats['precision_accuracy_gain']['low']>.03 and stats['precision_brier_reduction']['low']>.015},
        'G4_information_efficiency':{'pass':stats['ig_observation_reduction']['low']>4 and stats['ig_accuracy_change']['low']>-.02}}
 comp={'rows':len(rows),'expected':len(SEEDS)*len(RULES),'unique':len({(r['seed'],r['rule']) for r in rows})};comp['pass']=comp['rows']==comp['expected']==comp['unique']
 if not comp['pass']:raise RuntimeError(str(comp))
 result={'experiment':'III-B2','seeds':list(SEEDS),'preregistered_gates':GATES.strip(),'statistics':stats,'gates':gates,'completeness':comp}
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# 実験III-B2 結果：停止規則の独立効果','','## 仮説群','']
 for k,v in gates.items():lines.append(f"- **{k}**: {'PASS' if v['pass'] else 'FAIL'}")
 lines += ['','## 対応差','','|測定量|平均|95%区間|','|---|---:|---:|']
 for k,v in stats.items():lines.append(f"|{k}|{v['mean']:.6f}|[{v['low']:.6f}, {v['high']:.6f}]|")
 lines += ['','停止後も総計算回数は一致している。差は総計算時間ではなく、証拠取得を止める規則から生じた。']
 (out/'result_b2.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 prov={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'script_sha256':sha256(Path(__file__).resolve()),'preregistration_sha256':sha256(Path(__file__).parent/'preregistration_b2.md')}
 (out/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'gates':{k:v['pass'] for k,v in gates.items()},'completeness':comp,'output':str(out)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
