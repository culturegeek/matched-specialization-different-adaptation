"""Chronologically separated real sensor reconstruction with three-target state exchange."""
from pathlib import Path
import argparse,copy,json,hashlib,time,sys,platform
from dataclasses import asdict
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor/three_target'))
import run_adaptation as B
V=B.V
WEIGHTS=[1/3,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

@torch.no_grad()
def train_standardized_ridge_mse(z_train,y_train,z_test,y_test,ridge):
 x=torch.cat([z_train,torch.ones(len(z_train),1)],dim=1);xt=torch.cat([z_test,torch.ones(len(z_test),1)],dim=1);eye=torch.eye(x.shape[1]);eye[-1,-1]=0;w=torch.linalg.solve(x.T@x+ridge*eye,x.T@y_train)
 return float(((xt@w-y_test)**2).mean())
# Only the readout error normalization changes: sources are standardized once on training data.
V.ridge_error=train_standardized_ridge_mse

def splits_for(seed,phase):
 spec=json.loads((HERE/'data/split_manifest.json').read_text());src=HERE/'data/energydata_complete.csv';assert sha(src)==spec['source_csv_sha256'];d=pd.read_csv(src);mean=np.array(spec['normalization_mean_train_only']);std=np.array(spec['normalization_std_train_only']);train=d.iloc[:spec['splits']['train']['stop_row_exclusive']][spec['columns']].to_numpy(float);assert np.array_equal(mean,train.mean(0)) and np.array_equal(std,train.std(0));rng=np.random.default_rng(seed);rotation,_=np.linalg.qr(rng.normal(size=(6,6)))
 names=['train',('pilot' if phase in ['pilot','smoke'] else 'confirm')+'_probe',('pilot' if phase in ['pilot','smoke'] else 'confirm')+'_test'];out=[]
 for name in names:
  r=spec['splits'][name];x=(d.iloc[r['start_row']:r['stop_row_exclusive']][spec['columns']].to_numpy(float)-mean)/std;parts={t:x[:,2*i:2*i+2] for i,t in enumerate('ABC')};noisy={level:{t:(parts[t]+sig*rng.normal(size=parts[t].shape))/np.sqrt(1+sig*sig) for t in 'ABC'} for level,sig in [('high',.05),('low',1.2),('initial',.15)]}
  def mix(p):return torch.tensor(np.concatenate([p[t] for t in 'ABC'],axis=1)@rotation,dtype=torch.float32)
  inputs={t+'_precise':mix({k:noisy['high' if k==t else 'low'][k] for k in 'ABC'}) for t in 'ABC'};inputs.update({'all_high':mix(noisy['high']),'all_low':mix(noisy['low']),'initial':mix(noisy['initial'])});out.append(V.SignalSplit(inputs,{t:torch.tensor(parts[t],dtype=torch.float32) for t in 'ABC'}))
 return out,spec,names

def task(arg):
 phase,seed,cap,selected,out=arg;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);started=time.time();splits,spec,names=splits_for(seed,phase);cfg=V.Config(source_dim=2,train_size=len(splits[0].targets['A']),probe_size=len(splits[1].targets['A']),test_size=len(splits[2].targets['A']));initial=V.prepare_initial_state(cfg,cap,seed,splits[0],B.DEVICE);folder=Path(out)/f'capacity_{cap}'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=False)
 try:
  forms=[];endpoints={}
  for route in B.ROUTES:
   for f in 'ABC':
    ws=[1/3] if route=='acquisition' else WEIGHTS if phase=='pilot' else [selected[f]]
    for w in ws:
     pair=B.form_one(cfg,cap,seed,route,f,w,initial,splits[0]);met=B.metrics(B.make_model(cfg,cap,pair[0]),splits,cfg);forms.append({'seed':seed,'capacity':cap,'origin':f,'route':route,'weight':w,**met,**B.endpoint_vector(met,f)})
     if phase!='pilot':endpoints[route,f]=pair
  V.write_csv(folder/'formation_endpoints.csv',forms)
  sensor=V.sensor_diagnostics(cfg,splits[1],splits[2],B.DEVICE)
  for t in 'ABC':sensor['training_mean_baseline_D_'+t]=float((splits[2].targets[t]**2).mean())
  dump(folder/'sensor_diagnostics.json',sensor)
  if phase!='pilot':
   torch.save(endpoints,folder/'formation_checkpoints.pt');rows,states,norms=B.pulse(cfg,cap,seed,endpoints,splits);V.write_csv(folder/'timecourse.csv',rows);V.write_csv(folder/'state_diagnostics.csv',states);V.write_csv(folder/'update_diagnostics.csv',norms)
  dump(folder/'run.json',{'complete':True,'phase':phase,'seed':seed,'capacity':cap,'config':asdict(cfg),'selected_weights':selected if phase!='pilot' else None,'data_blocks':names,'seconds':time.time()-started});return seed,cap,round(time.time()-started,2)
 except Exception as e:dump(folder/'failure.json',{'error':repr(e)});raise

def frozen():return [HERE/'protocol.md',HERE/'run_natural.py',HERE/'analyze_natural.py',HERE/'calibration.json']+sorted((HERE/'data').glob('*'))+sorted((HERE/'vendor').rglob('*.py'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','smoke','lock','confirm'],required=True);ap.add_argument('--out',type=Path);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
 if a.phase=='lock':
  p=HERE/'registration.json';assert not p.exists();dump(p,{'utc':datetime.now(timezone.utc).isoformat(),'seeds':list(range(36000,36040)),'calibration':json.loads((HERE/'calibration.json').read_text()),'sha256':{str(p.relative_to(HERE)):sha(p) for p in frozen()},'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__});print('locked');return
 if a.phase=='confirm':
  reg=json.loads((HERE/'registration.json').read_text());assert all(sha(HERE/p)==h for p,h in reg['sha256'].items())
 assert a.out and not a.out.exists();a.out.mkdir(parents=True)
 seeds=range(35000,35008) if a.phase=='pilot' else [34999] if a.phase=='smoke' else range(36000,36040)
 cal=json.loads((HERE/'calibration.json').read_text()) if a.phase=='confirm' else None
 if a.phase=='pilot':dump(a.out/'pre_run.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':{str(p.relative_to(HERE)):sha(p) for p in [HERE/'protocol.md',HERE/'run_natural.py',HERE/'data/split_manifest.json',HERE/'data/energydata_complete.csv']+sorted((HERE/'vendor').rglob('*.py'))}})
 args=[(a.phase,s,c,{f:float(cal['selected'][str(c)][f]['weight']) for f in 'ABC'} if cal else {f:.65 for f in 'ABC'},str(a.out)) for c in [2,3] for s in seeds]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for s,c,t in pool.map(task,args):print(f'{a.phase} capacity={c} seed={s} seconds={t}',flush=True)
 dump(a.out/'manifest.json',{str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print('complete',a.phase,flush=True)
if __name__=='__main__':main()
