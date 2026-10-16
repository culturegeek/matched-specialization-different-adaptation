"""Independent reductions from checkpoint tensors and seed-level formation tables."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import torch
ap=argparse.ArgumentParser();ap.add_argument('--archive-root',type=Path,required=True);ap.add_argument('--result-root',type=Path,default=Path(__file__).parent/'saved_results');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
r=json.loads((a.result_root/'result.json').read_text());table=pd.read_csv(a.result_root/'checkpoint_states.csv');n=0
for name,h in r['input_sha256'].items():assert hashlib.sha256((a.archive_root/name).read_bytes()).hexdigest()==h
for (setting,seed),g in table.groupby(['setting','seed']):
 paths=list((a.archive_root/'experiments/iv-extension').glob(f'runs_*/{setting}/seed_{seed}/formation_checkpoints.pt'));assert len(paths)==1
 ep=torch.load(paths[0],weights_only=True,map_location='cpu')
 for row in g.itertuples():
  model,opt=ep[row.route,row.direction];pg=opt['param_groups'][0];pre={'all':'','encoder':'encoder.','encoder1':'encoder.0','encoder2':'encoder.2'}[row.layer];vs=0.;qs=0.;count=0
  for name,key in zip(model,pg['params']):
   if not name.startswith(pre):continue
   state=opt['state'][key];v=state['exp_avg_sq'].to(torch.float64);q=pg['lr']/(torch.sqrt(v/(1-pg['betas'][1]**float(state['step'])))+pg['eps']);vs+=float(v.sum());qs+=float(q.sum());count+=v.numel()
  assert np.isclose(vs,row.v_sum,rtol=1e-12,atol=1e-14);assert np.isclose(qs/count,row.inverse_scale,rtol=1e-12);n+=1
for beta in [.9,.99,.999]:
 short='base' if beta==.999 else f'beta{beta}_steps500';long=f'beta{beta}_steps1500'
 for layer in ['all','encoder','encoder1','encoder2']:
  for metric in ['v_sum','inverse_scale']:
   vals=[]
   for setting in [short,long]:
    q=table[(table.setting==setting)&(table.layer==layer)].pivot(index=['seed','direction'],columns='route',values=metric);vals.append(np.log(q.acquisition/q['update']).groupby(level='seed').mean())
   v=float(np.exp((vals[1]-vals[0]).mean()));assert abs(v-r['duration_pairs'][f'{beta}/{layer}'][metric+'_ratio_change']['mean'])<1e-12
f=pd.read_csv(a.archive_root/'experiments/iv-formation-logging/results/seed_final.csv')
for layer in ['encoder1','encoder2']:
 q=f[f.layer==layer].pivot(index='seed',columns='route',values='v_self');assert abs((q.acquisition-q['update']).mean()-r['self_differences'][layer]['mean'])<1e-15
x={'status':'PASS','checkpoint_layer_rows':n,'duration_mean_ratios':24,'paired_square_sum_differences':2,'input_hashes':len(r['input_sha256']),'method':'Independent torch reductions versus archived NumPy reductions; every duration ratio independently paired by seed. Intervals separately bound to source JSON.'};a.out.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
