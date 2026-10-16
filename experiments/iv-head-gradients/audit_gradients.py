"""Independent analytic chain-rule audit of saved gradient diagnostic CSVs."""
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np
import pandas as pd
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def vec(*parts):return np.concatenate([x.reshape(-1) for x in parts])
def manual(ms,x,y,head):
 w1,b1,w2,b2=[ms[k].double().numpy() for k in ['encoder.0.weight','encoder.0.bias','encoder.2.weight','encoder.2.bias']];wh,bh=[ms[f'head_{head}.{k}'].double().numpy() for k in ['weight','bias']]
 x=x.astype('float64');y=y.astype('float64');h=np.tanh(x@w1.T+b1);z=np.tanh(h@w2.T+b2);p=z@wh.T+bh;e=2*(p-y)/(len(x)*y.shape[1]);dz=e@wh;d2=dz*(1-z*z);d1=(d2@w2)*(1-h*h)
 return [vec(d1.T@x,d1.sum(0)),vec(d2.T@h,d2.sum(0))],vec(e.T@z,e.sum(0)),float(((p-y)**2).mean()),float(((dz*len(x))**2).sum(1).mean())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive-root',type=Path,required=True);ap.add_argument('--analysis-root',type=Path,default=Path(__file__).parent);a=ap.parse_args();sys.path.insert(0,str(a.archive_root/'experiments/iv-extension/vendor'));import run_followup as F
 torch.set_num_threads(1);d=a.analysis_root;raw=pd.read_csv(d/'results/batch_gradients.csv').set_index(['seed','direction','route','input','batch','layer']);checks=0;maxabs=0.;cfg=F.D.config_for_stage('confirm')
 for seed in range(20000,20040):
  ep=torch.load(a.archive_root/f'experiments/iv-followup/results_confirm/seed_{seed}/formation_checkpoints.pt',weights_only=True,map_location='cpu');_,_,test=F.B.make_splits(cfg,seed);batches=np.random.default_rng(820000+seed).integers(0,1536,(32,256))
  for direction in ['A','B']:
   inp={'both_high':test.x_both_high,'both_low':test.x_both_low,'favored_precise':test.x_a_precise if direction=='A' else test.x_b_precise,'other_precise':test.x_b_precise if direction=='A' else test.x_a_precise}
   for route in ['acquisition','update']:
    ms,_=ep[(route,direction)]
    for name,xt in inp.items():
     for batch,ix in enumerate(batches):
      f,fg,fl,fz=manual(ms,xt[ix].numpy(),test.a[ix].numpy() if direction=='A' else test.b[ix].numpy(),direction.lower());o,og,ol,oz=manual(ms,xt[ix].numpy(),test.b[ix].numpy() if direction=='A' else test.a[ix].numpy(),'b' if direction=='A' else 'a')
      for li,layer in enumerate(['encoder1','encoder2']):
       r=raw.loc[(seed,direction,route,name,batch,layer)];v={'favored_g2':f[li]@f[li],'other_g2':o[li]@o[li],'cross':f[li]@o[li],'favored_head_g2':fg@fg,'other_head_g2':og@og,'favored_loss':fl,'other_loss':ol,'favored_latent_g2':fz,'other_latent_g2':oz}
       for k,val in v.items():
        tol=2e-9+2e-5*(np.linalg.norm(f[li])*np.linalg.norm(o[li]) if k=='cross' else abs(val))
        assert abs(r[k]-val)<=tol,(seed,direction,route,name,layer,k,r[k],val,tol);maxabs=max(maxabs,abs(r[k]-val))
       checks+=1
 # Aggregate independently from batch CSV, using numpy arrays in fixed seed/direction order.
 res=json.loads((d/'results/result.json').read_text());idx=np.random.default_rng(2026101611).integers(0,40,(5000,40));ci_checks=0
 for inp in ['both_high','both_low','favored_precise','other_precise']:
  for layer in ['encoder1','encoder2']:
   for head in ['favored','other']:
    logs=[]
    for seed in range(20000,20040):
     vals=[]
     for dr in ['A','B']:
      aa=raw.xs((seed,dr,'acquisition',inp),level=['seed','direction','route','input']).xs(layer,level='layer')[head+'_g2'].to_numpy().mean();bb=raw.xs((seed,dr,'update',inp),level=['seed','direction','route','input']).xs(layer,level='layer')[head+'_g2'].to_numpy().mean();vals.append(.5*np.log(aa/bb))
     logs.append(np.mean(vals))
    logs=np.array(logs);est=np.exp([logs.mean(),*np.quantile(logs[idx].mean(1),[.025,.975])]);want=res['by_input'][inp][layer][head];assert np.allclose(est,[want[k] for k in ['mean','low','high']],rtol=1e-11,atol=1e-11);ci_checks+=1
 lock=json.loads((d/'analysis_plan_lock.json').read_text());assert all(sha(d/n)==h for n,h in lock['hashes'].items())
 inputs=json.loads((d/'results/input_hashes.json').read_text());assert all(sha(a.archive_root/n)==h for n,h in inputs.items() if not n.startswith('generated_'))
 out={'analytic_chain_rule_rows_passed':checks,'seed_ratio_intervals_passed':ci_checks,'max_absolute_float32_vs_float64_discrepancy':maxabs,'all_checkpoints_and_locked_sources_unchanged':True,'status':'PASS'};(d/'independent_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
