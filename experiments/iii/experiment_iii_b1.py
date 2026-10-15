#!/usr/bin/env python3
"""実験III-B1: 四要因による課題再編の確認実験。"""
from __future__ import annotations
import argparse,csv,itertools,json,platform,sys
from pathlib import Path
import numpy as np
from experiment_iii_a0 import sha256
from pilot_b1_task_reframing_v2 import run

SEEDS=tuple(range(3700,3740));FACTORS=list(itertools.product((0,1),(0,1),(1,10),(0.10,1.0)))
BOOT_SEED=20260824;BOOT_N=10000
GATES="""
G1 plasticity: all P=0 switch rates zero; P=1,Hpred=10,Heval=1 pooled switch-rate CI lower >0.10.
G2 horizons: under P=1, long-long health-frame fraction exceeds max other horizon cells by paired CI lower >0.20.
G3 exploration: under P=1,long-long, high-low alignment gain CI lower >0.25 and risk reduction CI lower >0.01.
G4 plasticity outcome: under E=1,long-long, P=1 versus P=0 risk reduction CI lower >0.05.
Unit=seed 3700--3739; percentile bootstrap seed=20260824, B=10000.
"""
def args():p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',default='results_b1');return p.parse_args()
def ci(x):
 x=np.asarray(x,float);rng=np.random.default_rng(BOOT_SEED);m=x[rng.integers(0,len(x),(BOOT_N,len(x)))].mean(1)
 return {'mean':float(x.mean()),'low':float(np.quantile(m,.025)),'high':float(np.quantile(m,.975))}
def main():
 a=args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
 for seed in SEEDS:
  for e,pk,hp,he in FACTORS:rows.append({'seed':seed,'explore':e,'plastic':pk,'Hpred':hp,'Heval':he,**run(seed,e,pk,hp,he)})
 with (out/'factorial.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 by={(r['seed'],r['explore'],r['plastic'],r['Hpred'],r['Heval']):r for r in rows}
 p0max=max(r['switch_rate'] for r in rows if r['plastic']==0)
 long_switch=[];horizon_gap=[];align_gain=[];explore_risk_gain=[];plastic_risk_gain=[]
 for s in SEEDS:
  long_switch.append(np.mean([by[(s,e,1,10,1.0)]['switch_rate'] for e in (0,1)]))
  ll=np.mean([by[(s,e,1,10,1.0)]['health_frame_fraction'] for e in (0,1)])
  other=max(np.mean([by[(s,e,1,hp,he)]['health_frame_fraction'] for e in (0,1)]) for hp,he in ((1,.1),(1,1.0),(10,.1)))
  horizon_gap.append(ll-other)
  hi=by[(s,1,1,10,1.0)];lo=by[(s,0,1,10,1.0)];fixed=by[(s,1,0,10,1.0)]
  align_gain.append(hi['frame_health_alignment']-lo['frame_health_alignment'])
  explore_risk_gain.append(lo['risk']-hi['risk'])
  plastic_risk_gain.append(fixed['risk']-hi['risk'])
 stats={'long_switch_rate':ci(long_switch),'horizon_interaction_gap':ci(horizon_gap),
        'exploration_alignment_gain':ci(align_gain),'exploration_risk_gain':ci(explore_risk_gain),
        'plasticity_risk_gain':ci(plastic_risk_gain)}
 gates={'G1_plasticity':{'pass':p0max==0 and stats['long_switch_rate']['low']>0.10,'p0_max':p0max},
        'G2_horizon_interaction':{'pass':stats['horizon_interaction_gap']['low']>0.20},
        'G3_exploration':{'pass':stats['exploration_alignment_gain']['low']>0.25 and stats['exploration_risk_gain']['low']>0.01},
        'G4_plasticity_outcome':{'pass':stats['plasticity_risk_gain']['low']>0.05}}
 comp={'rows':len(rows),'expected':len(SEEDS)*len(FACTORS),'unique':len({(r['seed'],r['explore'],r['plastic'],r['Hpred'],r['Heval']) for r in rows})};comp['pass']=comp['rows']==comp['expected']==comp['unique']
 if not comp['pass']:raise RuntimeError(str(comp))
 result={'experiment':'III-B1','seeds':list(SEEDS),'preregistered_gates':GATES.strip(),'statistics':stats,'gates':gates,'completeness':comp}
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# 実験III-B1 結果：課題再編の四要因分離','','## 仮説群','']
 for k,v in gates.items():lines.append(f"- **{k}**: {'PASS' if v['pass'] else 'FAIL'}")
 lines += ['','## 対応差','','|測定量|平均|95%区間|','|---|---:|---:|']
 for k,v in stats.items():lines.append(f"|{k}|{v['mean']:.6f}|[{v['low']:.6f}, {v['high']:.6f}]|")
 lines += ['','この実験の課題再編則は外部設計である。人間の開放性や課題志向を測ったものではない。']
 (out/'result_b1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 prov={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'script_sha256':sha256(Path(__file__).resolve()),'preregistration_sha256':sha256(Path(__file__).parent/'preregistration_b1.md')}
 (out/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'gates':{k:v['pass'] for k,v in gates.items()},'completeness':comp,'output':str(out)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
