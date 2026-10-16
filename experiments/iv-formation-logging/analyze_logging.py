"""Seed-paired descriptive summaries; no pass/fail threshold is selected."""
import argparse,csv,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
LAYERS=['encoder1','encoder2','head_a','head_b'];ROUTES=['acquisition','update'];WINDOWS=[(1,100),(101,250),(251,500),(1,500)]
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def write(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
 arrays=[];seeds=[]
 for folder in sorted(a.input.glob('seed_*')):
  data=np.load(folder/'records.npz');arrays.append(data['values']);seeds.append(int(folder.name.split('_')[-1]));fields=list(data['fields'])
 assert seeds==list(range(20000,20040)),seeds
 A=np.concatenate(arrays);ix={f:i for i,f in enumerate(fields)}
 rng=np.random.default_rng(2026101702);boot=rng.integers(0,40,size=(5000,40))
 def summary(v,log=False):
  v=np.array(v);z=np.log(v) if log else v;point=z.mean();lo,hi=np.quantile(z[boot].mean(axis=1),[.025,.975]);fn=np.exp if log else float
  return {'mean':float(fn(point)),'low':float(fn(lo)),'high':float(fn(hi)),'n':40}
 final=[];window=[];trajectory=[]
 for seed in seeds:
  for li,layer in enumerate(LAYERS):
   for ri,route in enumerate(ROUTES):
    rows=A[(A[:,0]==seed)&(A[:,ix['layer']]==li)&(A[:,ix['route']]==ri)]
    end=rows[rows[:,ix['step']]==500]
    d={'seed':seed,'layer':layer,'route':route}
    for field in ['v_after','v_initial','v_self','v_cross','v_regularization','v_rounding','inverse_scale']:
     d[field]=float(end[:,ix[field]].mean())
    d['cancellation']=float(np.mean(-end[:,ix['v_cross']]/end[:,ix['v_self']]))
    for field in ['v_initial','v_self','v_cross','v_regularization','v_rounding']:
     d[field+'_fraction']=float(np.mean(end[:,ix[field]]/end[:,ix['v_after']]))
    final.append(d)
    for lo,hi in WINDOWS:
     sel=rows[(rows[:,ix['step']]>=lo)&(rows[:,ix['step']]<=hi)]
     for precise in [-1,0,1]:
      r=sel if precise==-1 else sel[sel[:,ix['favored_precise']]==precise]
      if not len(r):continue
      wd={'seed':seed,'layer':layer,'route':route,'start':lo,'end':hi,'favored_precise':precise,'rows':len(r)}
      for field in ['favored_g2','other_g2','cosine','weighted_self','weighted_cross','task_g2','regularized_g2','update_g2']:
       wd[field]=float(r[:,ix[field]].mean())
      wd['cancellation']=-wd['weighted_cross']/wd['weighted_self'] if wd['weighted_self'] else 0
      window.append(wd)
 result={'status':'exploratory','n_seeds':40,'histories':160,'rows':len(A),'bootstrap_samples':5000,'bootstrap_seed':2026101702,'final':{},'windows':{}}
 for layer in LAYERS:
  by={route:[r for r in final if r['layer']==layer and r['route']==route] for route in ROUTES};entry={}
  for route in ROUTES:entry[route]={f:summary([r[f] for r in by[route]]) for f in by[route][0] if f not in ['seed','layer','route']}
  ratios=[]
  for seed in seeds:
   means=[]
   for di in [0,1]:
    pair=A[(A[:,0]==seed)&(A[:,1]==di)&(A[:,3]==500)&(A[:,4]==LAYERS.index(layer))]
    means.append(np.log(pair[pair[:,2]==0,ix['v_after']][0]/pair[pair[:,2]==1,ix['v_after']][0]))
   ratios.append(np.exp(np.mean(means)))
  entry['v_ratio']=summary(ratios,log=True);result['final'][layer]=entry
  for lo,hi in WINDOWS:
   for precise in [-1,0,1]:
    for route in ROUTES:
     rows=[r for r in window if r['layer']==layer and r['route']==route and r['start']==lo and r['end']==hi and r['favored_precise']==precise]
     if len(rows)!=40:continue
     result['windows'][f'{layer}/{route}/{lo}-{hi}/precision_{precise}']={f:summary([r[f] for r in rows]) for f in ['favored_g2','other_g2','cosine','cancellation','regularized_g2','update_g2']}
 for step in range(1,501):
  for li,layer in enumerate(LAYERS):
   row={'step':step,'layer':layer}
   for ri,route in enumerate(ROUTES):
    s=A[(A[:,3]==step)&(A[:,4]==li)&(A[:,2]==ri)]
    for field in ['v_after','v_self','v_cross','v_initial','cosine','regularized_g2']:
     row[route+'_'+field]=float(s[:,ix[field]].mean())
   row['v_ratio']=row['acquisition_v_after']/row['update_v_after']
   trajectory.append(row)
 write(a.out/'seed_final.csv',final);write(a.out/'seed_windows.csv',window);write(a.out/'trajectory.csv',trajectory);dump(a.out/'result.json',result)
 print(json.dumps({k:{'v_ratio':v['v_ratio'],'cancellation':{r:v[r]['cancellation'] for r in ROUTES}} for k,v in result['final'].items()},indent=2))
if __name__=='__main__':main()
