"""Exploratory endpoint gradient diagnostic; no optimizer update or refitting."""
from pathlib import Path
import argparse,sys,json,hashlib,platform,time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sq(x):return float(x.double().square().sum())
def dot(a,b):return float((a.double()*b.double()).sum())
def flat(gs):return torch.cat([g.detach().double().reshape(-1) for g in gs])
def interval(x,idx,exp=False):
 x=np.asarray(x);r=[x.mean(),*np.quantile(x[idx].mean(1),[.025,.975])]
 if exp:r=np.exp(r)
 return dict(zip(['mean','low','high'],map(float,r)))

def summarize(rows,out):
 df=pd.DataFrame(rows);df.to_csv(out/'batch_gradients.csv',index=False)
 keys=['seed','direction','route','input','layer'];metric=[c for c in df.columns if c not in keys+['batch']]
 c=df.groupby(keys)[metric].mean();c.to_csv(out/'checkpoint_gradients.csv')
 ratios=[];summary={};idx=np.random.default_rng(2026101611).integers(0,40,(5000,40))
 for inp in ['both_high','both_low','favored_precise','other_precise']:
  summary[inp]={}
  for layer in ['encoder1','encoder2']:
   d=c.xs((inp,layer),level=['input','layer']);p=d.unstack('route');summary[inp][layer]={}
   for head in ['favored','other']:
    e=head+'_g2';lr=.5*np.log(p[e]['acquisition']/p[e]['update']);lr=lr.groupby('seed').mean().sort_index();assert len(lr)==40
    summary[inp][layer][head]=interval(lr.to_numpy(),idx,exp=True)
    for seed,v in lr.items():ratios.append({'seed':int(seed),'input':inp,'layer':layer,'head':head,'log_rms_norm_ratio':v})
   for measure in ['favored_head_g2','other_head_g2','favored_latent_g2','other_latent_g2','favored_weight2','other_weight2','equal_total_g2','formation_total_g2']:
    lr=.5*np.log(p[measure]['acquisition']/p[measure]['update']);lr=lr.groupby('seed').mean().sort_index();summary[inp][layer][measure+'_ratio']=interval(lr.to_numpy(),idx,exp=True)
   for route in ['acquisition','update']:
    rd=d.xs(route,level='route').groupby('seed').mean();summary[inp][layer][route]={m:interval(rd[m].to_numpy(),idx) for m in ['cosine','favored_loss','other_loss','cross','equal_total_g2']}
 pd.DataFrame(ratios).to_csv(out/'seed_ratios.csv',index=False)
 return summary

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(a.archive_root/'experiments/iv-extension/vendor'));import run_followup as F
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);cfg=F.D.config_for_stage('confirm');hashes={};rows=[];checks={'checkpoint_unchanged':0,'direct_gradient_identity':0,'squared_gradient_identity':0};maximum={'direct_relative_error':0.,'squared_relative_error':0.}
 t=time.time()
 for seed in range(20000,20040):
  p=a.archive_root/f'experiments/iv-followup/results_confirm/seed_{seed}/formation_checkpoints.pt';hashes[str(p.relative_to(a.archive_root))]=sha(p);endpoints=torch.load(p,map_location='cpu',weights_only=True)
  _,_,test=F.B.make_splits(cfg,seed);batches=np.random.default_rng(820000+seed).integers(0,len(test.a),(32,256));hashes[f'generated_test_and_batches_seed_{seed}']=hashlib.sha256(batches.tobytes()+b''.join(v.numpy().tobytes() for v in vars(test).values())).hexdigest()
  for direction in ['A','B']:
   inputs={'both_high':test.x_both_high,'both_low':test.x_both_low,'favored_precise':test.x_a_precise if direction=='A' else test.x_b_precise,'other_precise':test.x_b_precise if direction=='A' else test.x_a_precise}
   for route in ['acquisition','update']:
    ms,os=endpoints[(route,direction)];model=F.B.make_model(cfg,F.DEVICE);model.load_state_dict(ms);params=list(model.parameters());enc=params[:4];pvec=[flat(enc[:2]),flat(enc[2:])]
    wt=.5 if route=='acquisition' else .8
    hf=getattr(model,'head_'+direction.lower());ho=getattr(model,'head_'+('b' if direction=='A' else 'a'))
    for inp,xall in inputs.items():
     for bi,ix in enumerate(batches):
      z,pa,pb=model(xall[ix]);lf=((pa-test.a[ix])**2).mean() if direction=='A' else ((pb-test.b[ix])**2).mean();lo=((pb-test.b[ix])**2).mean() if direction=='A' else ((pa-test.a[ix])**2).mean()
      gf=torch.autograd.grad(lf,enc+list(hf.parameters())+[z],retain_graph=True);go=torch.autograd.grad(lo,enc+list(ho.parameters())+[z],retain_graph=True)
      direct=torch.autograd.grad(wt*lf+(1-wt)*lo,enc)
      fvec=[flat(gf[:2]),flat(gf[2:4])];ovec=[flat(go[:2]),flat(go[2:4])];dvec=[flat(direct[:2]),flat(direct[2:])]
      for li,layer in enumerate(['encoder1','encoder2']):
       f,o=fvec[li],ovec[li];f2,o2,cr=sq(f),sq(o),dot(f,o);r=cfg.weight_decay*pvec[li]
       combined=wt*f+(1-wt)*o;reg=combined+r;err=float(torch.linalg.vector_norm(combined-dvec[li])/torch.linalg.vector_norm(dvec[li]).clamp_min(1e-20));maximum['direct_relative_error']=max(maximum['direct_relative_error'],err);assert err<1e-5;checks['direct_gradient_identity']+=1
       rebuilt=wt*wt*f2+(1-wt)**2*o2+2*wt*(1-wt)*cr+sq(r)+2*dot(combined,r);err2=abs(rebuilt-sq(reg))/max(sq(reg),1e-30);maximum['squared_relative_error']=max(maximum['squared_relative_error'],err2);assert err2<1e-10;checks['squared_gradient_identity']+=1
       rows.append({'seed':seed,'direction':direction,'route':route,'input':inp,'batch':bi,'layer':layer,'favored_g2':f2,'other_g2':o2,'cross':cr,'cosine':cr/np.sqrt(f2*o2),'equal_total_g2':sq(.5*f+.5*o),'formation_total_g2':sq(combined),'regularized_total_g2':sq(reg),'decay_g2':sq(r),'decay_cross':2*dot(combined,r),'favored_head_g2':sq(flat(gf[4:6])),'other_head_g2':sq(flat(go[4:6])),'favored_latent_g2':float((gf[-1].double()*len(ix)).square().sum(1).mean()),'other_latent_g2':float((go[-1].double()*len(ix)).square().sum(1).mean()),'favored_weight2':sq(hf.weight),'other_weight2':sq(ho.weight),'favored_loss':float(lf.detach()),'other_loss':float(lo.detach())})
    assert all(torch.equal(model.state_dict()[k],v) for k,v in ms.items());assert sha(p)==hashes[str(p.relative_to(a.archive_root))];checks['checkpoint_unchanged']+=1
  if seed%5==4:print('seeds complete',seed-19999,'elapsed',round(time.time()-t,1),flush=True)
 summary=summarize(rows,a.out)
 dump(a.out/'result.json',{'status':'exploratory_endpoint_gradients','seeds':40,'checkpoints':160,'batches_per_checkpoint_input':32,'rows':len(rows),'ratio':'geometric mean acquisition/update RMS batch-gradient norm; directions averaged in log space within seed','intervals':'individual 95% paired-seed bootstrap; 5000 resamples','by_input':summary})
 dump(a.out/'input_hashes.json',hashes);dump(a.out/'validation.json',{'checks':checks,'maximum_errors':maximum,'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'threads':1,'cpu':True,'elapsed_seconds':time.time()-t,'source_sha256':sha(Path(__file__)),'protocol_sha256':sha(HERE/'protocol.md')})
 print(json.dumps(summary['both_high'],indent=2),flush=True)
if __name__=='__main__':main()
