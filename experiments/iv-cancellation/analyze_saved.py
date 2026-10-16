"""Exploratory paired summaries of preserved trajectories and nine endpoint settings."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 rng=np.random.default_rng(2026101703)
 def summary(x,log=False):
  x=np.asarray(x,float);z=np.log(x) if log else x;b=z[rng.integers(len(z),size=(10000,len(z)))].mean(1);f=np.exp if log else lambda y:y
  return dict(mean=float(f(z.mean())),low=float(f(np.quantile(b,.025))),high=float(f(np.quantile(b,.975))),n=len(x))
 root=a.archive_root;inputs={};r={'status':'exploratory existing-data analysis','bootstrap_seed':2026101703,'resamples':10000,'self_differences':{},'settings':{},'duration_pairs':{},'observation_strata':{}}
 fp=root/'experiments/iv-formation-logging/results/seed_final.csv';df=pd.read_csv(fp);inputs[str(fp.relative_to(root))]=sha(fp)
 for layer in ['encoder1','encoder2']:
  x=df[df.layer==layer].pivot(index='seed',columns='route',values='v_self');r['self_differences'][layer]=summary(x.acquisition-x['update'])
 fp=root/'experiments/iv-formation-logging/results/seed_windows.csv';df=pd.read_csv(fp);inputs[str(fp.relative_to(root))]=sha(fp)
 for (layer,route,lo,hi,prec),g in df.groupby(['layer','route','start','end','favored_precise']):
  if layer in ['encoder1','encoder2']:
   r['observation_strata'][f'{layer}/{route}/{lo}-{hi}/{prec}']={k:summary(g[k]) for k in ['cosine','cancellation']}
 rows=[]
 for sub in ['runs_general','runs_beta']:
  for file in sorted((root/'experiments/iv-extension'/sub).glob('*/seed_*/formation_checkpoints.pt')):
   name=file.parent.parent.name;seed=int(file.parent.name.split('_')[-1]);ep=torch.load(file,map_location='cpu',weights_only=True);inputs[str(file.relative_to(root))]=sha(file)
   cfg=json.loads((file.parent/'run.json').read_text())['config']
   for direction in ['A','B']:
    for route in ['acquisition','update']:
     ms,os=ep[route,direction];pg=os['param_groups'][0];names=list(ms);keys=pg['params']
     for layer,prefix in [('all',''),('encoder','encoder.'),('encoder1','encoder.0'),('encoder2','encoder.2')]:
      states=[os['state'][k] for n,k in zip(names,keys) if n.startswith(prefix)];v=np.concatenate([q['exp_avg_sq'].double().numpy().ravel() for q in states]);q=np.concatenate([pg['lr']/(np.sqrt(s['exp_avg_sq'].double().numpy().ravel()/(1-pg['betas'][1]**float(s['step'])))+pg['eps']) for s in states])
      rows.append(dict(setting=name,seed=seed,direction=direction,route=route,layer=layer,beta2=cfg['beta2'],steps=cfg['steps'],v_sum=v.sum(),inverse_scale=q.mean()))
 df=pd.DataFrame(rows);df.to_csv(a.out/'checkpoint_states.csv',index=False);ratios={}
 for (name,layer),g in df.groupby(['setting','layer']):
  ratios[name,layer]={};r['settings'].setdefault(name,{})[layer]={}
  for metric in ['v_sum','inverse_scale']:
   w=g.pivot(index=['seed','direction'],columns='route',values=metric);z=np.log(w.acquisition/w['update']).groupby('seed').mean();ratios[name,layer][metric]=z
   r['settings'][name][layer][metric+'_ratio']=summary(np.exp(z),True)
 for beta in [.9,.99,.999]:
  short='base' if beta==.999 else f'beta{beta}_steps500';long=f'beta{beta}_steps1500'
  for layer in ['all','encoder','encoder1','encoder2']:
   r['duration_pairs'][f'{beta}/{layer}']={metric+'_ratio_change':summary(np.exp(ratios[long,layer][metric]-ratios[short,layer][metric]),True) for metric in ['v_sum','inverse_scale']}
 r['input_sha256']=inputs;(a.out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['self_differences','duration_pairs']},indent=2))
if __name__=='__main__':main()
