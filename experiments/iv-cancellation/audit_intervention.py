"""Independent analytic gradients, projection algebra, and raw-data checks."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from numerical_audit import TensorAudit
from run_intervention import parent,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive-root',type=Path,required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--tolerance',type=float,default=0.0);ap.add_argument('--seeds',type=int,nargs='+');a=ap.parse_args();numeric=TensorAudit(a.tolerance);torch.set_num_threads(1);F=parent(a.archive_root);cfg=F.D.config_for_stage('confirm');dev=torch.device('cpu');s=importlib.util.spec_from_file_location('analytic',a.archive_root/'experiments/iv-head-gradients/audit_gradients.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M);n=0;rows=0;starts=0;maxdiff=0.;baseline=0
 seeds=sorted(a.seeds) if a.seeds else list(range(38000,38040));assert len(set(seeds))==len(seeds) and set(seeds)<=set(range(38000,38040));folders=[a.input/f'seed_{seed}' for seed in seeds];assert all(f.is_dir() for f in folders)
 for folder in folders:
  seed=int(folder.name.split('_')[-1]);splits=F.B.make_splits(cfg,seed);train=splits[0];tr=pd.read_csv(folder/'formation_trace.csv');tc=pd.read_csv(folder/'timecourse.csv');ep=torch.load(folder/'formation_checkpoints.pt',map_location='cpu',weights_only=False)
  assert len(tr)==3000 and len(tc)==130;assert np.isfinite(tr.select_dtypes(include='number')).all().all();rows+=len(tr)
  for (dr,rc),g in tc[tc.step==0].groupby(['direction','recipient']):
   for col in ['aligned_s','target_D','other_D','error_sum','aligned_low_s']:assert g[col].nunique()==1
   starts+=1
  nm,no=F.D.prepare_neutral(cfg,seed,train,dev)
  for dr in ['A','B']:
   for cond in ['acquisition','update','projected']:
    route='update' if cond=='update' else 'acquisition';acq,w=F.D.formation_condition(route,dr,.8);snaps=torch.load(folder/f'snapshots_{cond}_{dr}.pt',weights_only=False,map_location='cpu');[numeric.compare(nm[k],snaps[0]['model'][k],'weights',(seed,dr,cond,'ancestor',k)) for k in nm]
    if cond!='projected':
     want=F.D.form_state(cfg,seed,route,dr,.8,nm,no,train,dev);actual=ep[cond,dr];same=[numeric.compare(want[0][k],actual[0][k],'weights',(seed,dr,cond,k)) for k in want[0]]
     for k,state in want[1]['state'].items():
      assert torch.equal(state['step'],actual[1]['state'][k]['step'])
      for field,kind in [('exp_avg','m'),('exp_avg_sq','v')]:same.append(numeric.compare(state[field],actual[1]['state'][k][field],kind,(seed,dr,cond,k,field)))
     baseline+=int(all(same))
    indices=np.random.default_rng(600000+seed).integers(0,len(train.a),(500,256));schedule=F.D.balanced_schedule(500,seed,610000)
    for snap in snaps[:-1]:
     step=snap['step'];idx=indices[step];x=F.D.acquisition_tensor(train,acq,int(schedule[step]))[idx].numpy();ga=M.manual(snap['model'],x,train.a[idx].numpy(),'a')[0];gb=M.manual(snap['model'],x,train.b[idx].numpy(),'b')[0];u=w*np.concatenate(ga);v=(1-w)*np.concatenate(gb);dot=u@v
     if cond=='projected' and dot<0:up=u-dot/(v@v)*v;vp=v-dot/(u@u)*u
     else:up,vp=u,v
     val=dict(pre_self=u@u+v@v,pre_cross=2*dot,post_self=up@up+vp@vp,post_cross=2*(up@vp),pre_norm2=(u+v)@(u+v),post_norm2=(up+vp)@(up+vp));row=tr[(tr.direction==dr)&(tr.condition==cond)&(tr.step==step+1)].iloc[0]
     for k,value in val.items():
      tol=2e-9+5e-5*max(abs(value),val['pre_self']);diff=abs(row[k]-value);assert diff<=tol,(seed,cond,dr,step,k,diff,tol);maxdiff=max(maxdiff,diff)
     # Independently advance the scalar v sum from the preserved state.
     reg=up+vp+cfg.weight_decay*np.concatenate([snap['model'][k].double().numpy().ravel() for k in snap['model'] if k.startswith('encoder.')]);os=snap['optimizer'];old=sum(float(os['state'][i]['exp_avg_sq'].double().sum()) for k,i in zip(snap['model'],os['param_groups'][0]['params']) if k.startswith('encoder.'));pred=.999*old+.001*(reg@reg);assert abs(row.v_encoder-pred)<1e-10+5e-5*max(abs(pred),1e-12);n+=1
    if cond=='projected':
     g=tr[(tr.direction==dr)&(tr.condition==cond)];assert (g.post_cross>=-1e-7*g.post_self-1e-12).all()
   for rc,donor in [('acquisition','projected'),('projected','acquisition')]:
    own=ep[rc,dr][1];other=ep[donor,dr][1];out=F.adjusted_state(own,other,'v_other',1.)
    for k in own['state']:
     assert torch.equal(out['state'][k]['exp_avg'],own['state'][k]['exp_avg']) and torch.equal(out['state'][k]['step'],own['state'][k]['step']) and torch.equal(out['state'][k]['exp_avg_sq'],other['state'][k]['exp_avg_sq'])
 reg=json.loads((Path(__file__).parent/'registration.json').read_text());assert all(sha(Path(__file__).parent/f)==h for f,h in reg['files'].items())
 result=dict(status='PASS',seeds=seeds,full_cohort=seeds==list(range(38000,38040)),numerical_comparison={**numeric.report(),'torch':torch.__version__},baseline_endpoints_checked=len(seeds)*4,formation_rows=rows,analytic_gradient_and_v_checks=n,baseline_bitwise_endpoints=baseline,pulse_identical_recipient_starts=starts,maximum_analytic_difference=maxdiff,locked_files_unchanged=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
