"""Independent NumPy chain-rule and scalar-recursion audit of formation logs."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from run_logging import load_parent,sha,LAYERS

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive-root',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(1);F=load_parent(a.archive_root);cfg=F.D.config_for_stage('confirm')
 spec=importlib.util.spec_from_file_location('manual_chain_rule',a.archive_root/'experiments/iv-head-gradients/audit_gradients.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
 checked=0;analytic=0;max_error=0.;max_round_frac=0.;endpoint_checks=0;recur_rel=0.
 for seed in range(20000,20040):
  folder=a.input/f'seed_{seed}';raw=np.load(folder/'records.npz');A=raw['values'];ix={f:i for i,f in enumerate(raw['fields'])};assert A.shape==(8000,28);assert np.isfinite(A).all()
  train=F.B.make_splits(cfg,seed)[0];ep=torch.load(a.archive_root/f'experiments/iv-followup/results_confirm/seed_{seed}/formation_checkpoints.pt',weights_only=False,map_location='cpu')
  for di,dr in enumerate(['A','B']):
   for ri,route in enumerate(['acquisition','update']):
    s=np.load(folder/f'snapshots_{route}_{dr}.npz');base=F.B.make_model(cfg,torch.device('cpu'));names=[n for n,p in base.named_parameters()];shapes=[p.shape for p in base.parameters()];edges=np.cumsum([0]+[p.numel() for p in base.parameters()])
    ref=ep[(route,dr)];flat=lambda ts:torch.cat([t.reshape(-1) for t in ts]).numpy()
    for key,want in [('weights',flat(list(ref[0].values()))),('m',flat([v['exp_avg'] for v in ref[1]['state'].values()])),('v',flat([v['exp_avg_sq'] for v in ref[1]['state'].values()]))]:assert np.array_equal(s[key][-1],want),(seed,route,dr,key)
    endpoint_checks+=1
    assert np.array_equal(s['sample_indices'],np.random.default_rng(600000+seed).integers(0,len(train.a),(500,256)))
    assert np.array_equal(s['schedule'],F.D.balanced_schedule(500,seed,610000))
    acq,w=F.D.formation_condition(route,dr,.8);wf=w if dr=='A' else 1-w;wo=1-wf
    for si,step in enumerate(s['steps'][:-1]):
     ms={n:torch.from_numpy(s['weights'][si,edges[k]:edges[k+1]].reshape(shapes[k]).copy()) for k,n in enumerate(names)}
     idx=s['sample_indices'][step].astype(int);x=F.D.acquisition_tensor(train,acq,int(s['schedule'][step]))[idx].numpy()
     f,fh,_,_=M.manual(ms,x,(train.a if dr=='A' else train.b)[idx].numpy(),dr.lower());o,oh,_,_=M.manual(ms,x,(train.b if dr=='A' else train.a)[idx].numpy(),'b' if dr=='A' else 'a')
     f=f+[fh if dr=='A' else np.zeros_like(oh),fh if dr=='B' else np.zeros_like(oh)];o=o+[oh if dr=='B' else np.zeros_like(fh),oh if dr=='A' else np.zeros_like(fh)]
     for li in range(4):
      row=A[(A[:,1]==di)&(A[:,2]==ri)&(A[:,3]==step+1)&(A[:,4]==li)][0];g=wf*f[li]+wo*o[li];fv=f[li]@f[li];ov=o[li]@o[li];dot=f[li]@o[li]
      for field,value in [('favored_g2',fv),('other_g2',ov),('dot',dot),('weighted_self',wf*wf*fv+wo*wo*ov),('weighted_cross',2*wf*wo*dot),('task_g2',g@g)]:
       scale=np.sqrt(fv*ov) if field=='dot' else abs(value);tol=2e-9+3e-5*scale
       assert abs(row[ix[field]]-value)<=tol,(seed,dr,route,step,li,field,row[ix[field]],value,tol)
       max_error=max(max_error,abs(row[ix[field]]-value))
      analytic+=1
    for li in range(4):
     rows=A[(A[:,1]==di)&(A[:,2]==ri)&(A[:,4]==li)];assert np.array_equal(rows[:,3],np.arange(1,501));last=None
     for r in rows:
      if last is not None:assert r[ix['v_before']]==last[ix['v_after']]
      delta=r[ix['v_after']]-(.999*r[ix['v_before']]+.001*r[ix['regularized_g2']]);scale=r[ix['v_after']]+.001*r[ix['regularized_g2']]
      assert abs(delta)<=1e-7*scale+1e-12,(seed,dr,route,li,delta,scale);recur_rel=max(recur_rel,abs(delta)/(scale+1e-30))
      assert abs(r[ix['v_after']]-sum(r[ix[x]] for x in ['v_initial','v_self','v_cross','v_regularization','v_rounding']))<1e-10
      if last is not None:
       for cumulative,current in [('v_self','weighted_self'),('v_cross','weighted_cross'),('v_regularization','regularization')]:
        assert abs(r[ix[cumulative]]-(.999*last[ix[cumulative]]+.001*r[ix[current]]))<1e-10
      checked+=1;last=r
     max_round_frac=max(max_round_frac,abs(last[ix['v_rounding']])/last[ix['v_after']])
 lock=json.loads((Path(__file__).parent/'analysis_plan_lock.json').read_text());assert all(sha(Path(__file__).parent/n)==h for n,h in lock['files'].items())
 result={'status':'PASS','layer_step_records':checked,'analytic_chain_rule_records':analytic,'bitwise_endpoint_checks':endpoint_checks,'max_analytic_float32_vs_float64_difference':max_error,'max_relative_v_recursion_discrepancy':recur_rel,'max_final_rounding_fraction':max_round_frac,'plan_unchanged':True}
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
