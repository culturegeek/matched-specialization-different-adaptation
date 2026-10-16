"""Three-target endpoint calibration and six-direction state-exchange experiments."""
from pathlib import Path
import argparse,copy,hashlib,json,time,sys,platform
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
import experiment_v_a as V
TARGETS=V.TARGETS
ROUTES=['acquisition','update']
MODES=['own','m_other','v_other','reset','v_equal_scale','v_other_equal_scale']
WEIGHTS=[.55,.60,.65,.70,.75,.80,.85,.90,.95]
RECORD=[0,1,5,10,20,50]
DEVICE=torch.device('cpu')
TOLERANCES={'D0':.02,'D1':.02,'D2':.02,'error_sum':.02,'s01':.02,'s02':.02,'low_s01':.01,'low_s02':.01}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def configure():return V.Config()
def weight(f,w):return {t:w if t==f else (1-w)/2 for t in TARGETS}
def make_model(cfg,cap,ms=None):
 m=V.make_model(cfg,cap,DEVICE)
 if ms is not None:m.load_state_dict(ms)
 return m

def optimizer(m,cfg,state=None):
 o=torch.optim.Adam(m.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay,betas=(.9,.999),eps=1e-8)
 if state is not None:o.load_state_dict(copy.deepcopy(state))
 return o

@torch.no_grad()
def metrics(m,splits,cfg):
 _,p,t=splits;out={}
 for level in ['high','low']:
  d,q=V.representation_metrics(m,p.inputs['all_'+level],t.inputs['all_'+level],p,t,cfg,DEVICE)
  out.update({f'{level}_D_{x}':d[x] for x in TARGETS});out.update({f'{level}_q_{x}':q[x] for x in TARGETS});out[level+'_error_sum']=sum(d.values())
  for i in TARGETS:
   for j in TARGETS:
    if i!=j:out[f'{level}_s_{i}{j}']=(d[j]-d[i])/(d[j]+d[i])
 return out

def endpoint_vector(met,f):
 i=TARGETS.index(f);f,a,b=[TARGETS[(i+j)%3] for j in range(3)]
 return {'D0':met['high_D_'+f],'D1':met['high_D_'+a],'D2':met['high_D_'+b],'error_sum':met['high_error_sum'],'s01':met['high_s_'+f+a],'s02':met['high_s_'+f+b],'low_s01':met['low_s_'+f+a],'low_s02':met['low_s_'+f+b]}

def form_one(cfg,cap,seed,route,f,w,initial,train):
 m=make_model(cfg,cap,initial[0]);o=optimizer(m,cfg,initial[1]);rng=np.random.default_rng(300000+seed);schedule=V.balanced_acquisition_schedule(cfg,seed);n=len(train.targets['A'])
 for step in range(cfg.formation_steps):
  idx=torch.from_numpy(rng.integers(0,n,cfg.batch_size));acq=f if route=='acquisition' else TARGETS[int(schedule[step])];weights={t:1/3 for t in TARGETS} if route=='acquisition' else weight(f,w)
  V.train_weighted(m,o,train.inputs[acq+'_precise'][idx],{t:train.targets[t][idx] for t in TARGETS},weights)
 return copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict())

def q_mean(state,scale=1.):
 g=state['param_groups'][0];b=g['betas'][1]
 return float(torch.cat([g['lr']/(torch.sqrt(s['exp_avg_sq'].double().reshape(-1)*scale/(1-b**float(s['step'])))+g['eps']) for s in state['state'].values()]).mean())

def adjusted(own,other,mode,target):
 if mode=='reset':return None
 s=copy.deepcopy(own)
 if mode=='m_other':
  for k in s['state']:s['state'][k]['exp_avg']=other['state'][k]['exp_avg'].clone()
 if mode in ['v_other','v_other_equal_scale']:
  for k in s['state']:s['state'][k]['exp_avg_sq']=other['state'][k]['exp_avg_sq'].clone()
 if mode in ['v_equal_scale','v_other_equal_scale']:
  lo,hi=-30.,30.
  for _ in range(80):
   mid=(lo+hi)/2
   if q_mean(s,np.exp(mid))>target:lo=mid
   else:hi=mid
  for v in s['state'].values():v['exp_avg_sq'].mul_(np.exp((lo+hi)/2))
  assert abs(q_mean(s)/target-1)<2e-6
 for k in s['state']:
  assert torch.equal(s['state'][k]['step'],own['state'][k]['step'])
  if mode!='m_other':assert torch.equal(s['state'][k]['exp_avg'],own['state'][k]['exp_avg'])
  if mode in ['own','m_other']:assert torch.equal(s['state'][k]['exp_avg_sq'],own['state'][k]['exp_avg_sq'])
 return s

def pulse(cfg,cap,seed,endpoints,splits):
 train,_,_=splits;rows=[];states=[];norms=[]
 for f in TARGETS:
  target=float(np.sqrt(np.prod([q_mean(endpoints[r,f][1]) for r in ROUTES])))
  for dest in TARGETS:
   if dest==f:continue
   for mode in MODES:
    for route in ROUTES:
     ms,os=endpoints[route,f];other=endpoints['update' if route=='acquisition' else 'acquisition',f][1];m=make_model(cfg,cap,ms);state=adjusted(os,other,mode,target);o=optimizer(m,cfg,state)
     assert all(torch.equal(m.state_dict()[k],v) for k,v in ms.items())
     ident={'seed':seed,'capacity':cap,'origin':f,'destination':dest,'route':route,'mode':mode}
     states.append({**ident,'q_own':q_mean(os),'q_common_target':target,'q_initial':q_mean(state) if state else 0.})
     rng=np.random.default_rng(700000+seed);path=0.
     for step in range(51):
      if step in RECORD:rows.append({**ident,'step':step,'path_length':path,**metrics(m,splits,cfg)})
      if step==50:break
      idx=torch.from_numpy(rng.integers(0,len(train.targets['A']),cfg.batch_size));before=[p.detach().clone() for p in m.parameters()]
      V.train_weighted(m,o,train.inputs[dest+'_precise'][idx],{t:train.targets[t][idx] for t in TARGETS},weight(dest,.8))
      un=float(torch.sqrt(sum(((p.detach()-b).double()**2).sum() for p,b in zip(m.parameters(),before))));assert np.isfinite(un);path+=un
      norms.append({**ident,'step':step+1,'update_norm':un})
 return rows,states,norms

def task(arg):
 phase,seed,cap,selected,out=arg;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();cfg=configure();splits=V.make_splits(cfg,seed);initial=V.prepare_initial_state(cfg,cap,seed,splits[0],DEVICE);folder=Path(out)/f'capacity_{cap}'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=False)
 try:
  forms=[];endpoints={}
  for route in ROUTES:
   ws=[1/3] if route=='acquisition' else WEIGHTS if phase=='pilot' else [selected]
   for w in ws:
    for f in TARGETS:
     pair=form_one(cfg,cap,seed,route,f,w,initial,splits[0]);m=make_model(cfg,cap,pair[0]);met=metrics(m,splits,cfg)
     forms.append({'seed':seed,'capacity':cap,'origin':f,'route':route,'weight':w,**met,**endpoint_vector(met,f)})
     if phase!='pilot':endpoints[route,f]=pair
  V.write_csv(folder/'formation_endpoints.csv',forms)
  if phase!='pilot':
   torch.save(endpoints,folder/'formation_checkpoints.pt');rows,states,norms=pulse(cfg,cap,seed,endpoints,splits);V.write_csv(folder/'timecourse.csv',rows);V.write_csv(folder/'state_diagnostics.csv',states);V.write_csv(folder/'update_diagnostics.csv',norms)
  dump(folder/'run.json',{'complete':True,'phase':phase,'seed':seed,'capacity':cap,'seconds':time.time()-start,'config':asdict(cfg),'selected_weight':selected if phase!='pilot' else None})
  return seed,cap,round(time.time()-start,2)
 except Exception as e:dump(folder/'failure.json',{'error':repr(e)});raise

def frozen():return [HERE/'pilot_protocol.md',HERE/'run_adaptation.py',HERE/'analyze_adaptation.py',HERE/'calibration.json',HERE/'vendor/experiment_v_a.py']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','smoke','lock','confirm'],required=True);ap.add_argument('--out',type=Path);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
 if a.phase=='lock':
  p=HERE/'registration.json';assert not p.exists();dump(p,{'utc':datetime.now(timezone.utc).isoformat(),'confirm_seeds':list(range(34000,34040)),'calibration':json.loads((HERE/'calibration.json').read_text()),'sha256':{str(p.relative_to(HERE)):sha(p) for p in frozen()},'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__});print('locked');return
 if a.phase=='confirm':
  reg=json.loads((HERE/'registration.json').read_text());assert all(sha(HERE/k)==v for k,v in reg['sha256'].items())
 assert a.out and not a.out.exists();a.out.mkdir(parents=True)
 seeds=range(33000,33008) if a.phase=='pilot' else [32999] if a.phase=='smoke' else range(34000,34040)
 cal=json.loads((HERE/'calibration.json').read_text()) if a.phase=='confirm' else None
 if a.phase=='pilot':dump(a.out/'pre_run.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':{str(p.relative_to(HERE)):sha(p) for p in [HERE/'pilot_protocol.md',HERE/'run_adaptation.py',HERE/'vendor/experiment_v_a.py']}})
 args=[(a.phase,s,c,float(cal['selected'][str(c)]['weight']) if cal else .8,str(a.out)) for c in [2,3] for s in seeds]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for s,c,t in pool.map(task,args):print(f'{a.phase} capacity={c} seed={s} seconds={t}',flush=True)
 dump(a.out/'manifest.json',{str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print('complete',a.phase,flush=True)
if __name__=='__main__':main()
