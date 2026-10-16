"""Shared-encoder PCGrad formation and crossed second-moment transplants."""
import argparse,copy,hashlib,importlib.util,json,platform,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
SEEDS=list(range(38000,38040));SNAP=[0,100,250,500]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def parent(root):
 p=Path(root)/'experiments/iv-followup/run_followup.py';s=importlib.util.spec_from_file_location('pcgrad_parent',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def flat(ts):return torch.cat([x.detach().reshape(-1) for x in ts])
def project(u,v):
 dot=torch.dot(u,v)
 if float(dot)<0:
  return u-dot/torch.dot(v,v)*v,v-dot/torch.dot(u,u)*u
 return u.clone(),v.clone()
def form(F,cfg,seed,direction,condition,nm,no,train,out):
 B,D=F.B,F.D;dev=torch.device('cpu');m=B.make_model(cfg,dev);m.load_state_dict(nm);o=B.clone_optimizer(m,cfg,no);params=list(m.parameters());names=[n for n,p in m.named_parameters()];enc=[i for i,n in enumerate(names) if n.startswith('encoder.')];route='update' if condition=='update' else 'acquisition';acq,w=D.formation_condition(route,direction,.8);rng=np.random.default_rng(600000+seed);schedule=D.balanced_schedule(500,seed,610000);rows=[];snapshots=[]
 def snap(t):snapshots.append(dict(step=t,model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(o.state_dict())))
 snap(0)
 for step in range(1,501):
  idx=torch.from_numpy(rng.integers(0,len(train.a),cfg.batch_size));x=D.acquisition_tensor(train,acq,int(schedule[step-1]))[idx];o.zero_grad(set_to_none=True);_,pa,pb=m(x);la=((pa-train.a[idx])**2).mean();lb=((pb-train.b[idx])**2).mean()
  ga=torch.autograd.grad(la,params,retain_graph=True,allow_unused=True);gb=torch.autograd.grad(lb,params,retain_graph=True,allow_unused=True);ga=[torch.zeros_like(p) if g is None else g for g,p in zip(ga,params)];gb=[torch.zeros_like(p) if g is None else g for g,p in zip(gb,params)]
  u=flat([w*ga[i] for i in enc]);v=flat([(1-w)*gb[i] for i in enc]);up,vp=project(u,v) if condition=='projected' else (u,v);(w*la+(1-w)*lb).backward();before=flat(params).clone();head_before=[params[i].grad.clone() for i in range(len(params)) if i not in enc]
  if condition=='projected':
   joined=up+vp;pos=0
   for i in enc:n=params[i].numel();params[i].grad.copy_(joined[pos:pos+n].view_as(params[i]));pos+=n
  assert all(torch.equal(g,params[i].grad) for g,i in zip(head_before,[i for i in range(len(params)) if i not in enc]))
  o.step();d=dict(seed=seed,direction=direction,condition=condition,step=step,conflict=float(torch.dot(u,v))<0,pre_self=float(u.double().square().sum()+v.double().square().sum()),pre_cross=float(2*torch.dot(u.double(),v.double())),post_self=float(up.double().square().sum()+vp.double().square().sum()),post_cross=float(2*torch.dot(up.double(),vp.double())),pre_norm2=float((u.double()+v.double()).square().sum()),post_norm2=float((up.double()+vp.double()).square().sum()),v_encoder=float(sum(o.state[params[i]]['exp_avg_sq'].double().sum() for i in enc)),update_norm=float(torch.linalg.vector_norm(flat(params).double()-before.double())))
  assert np.isfinite(list(d[k] for k in ['pre_self','pre_cross','post_self','post_cross','v_encoder','update_norm'])).all();rows.append(d)
  if step in SNAP:snap(step)
 torch.save(snapshots,out/f'snapshots_{condition}_{direction}.pt');return (copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict())),rows

def run_one(arg):
 root,out,seed=arg;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);F=parent(root);D,B=F.D,F.B;cfg=D.config_for_stage('confirm');dev=torch.device('cpu');splits=B.make_splits(cfg,seed);nm,no=D.prepare_neutral(cfg,seed,splits[0],dev);folder=Path(out)/f'seed_{seed}';folder.mkdir(exist_ok=False);ep={};logs=[];forms=[];state=[];traces=[];norms=[];started=time.time()
 for direction in D.DIRECTIONS:
  for condition in ['acquisition','update','projected']:
   pair,rows=form(F,cfg,seed,direction,condition,nm,no,splits[0],folder);ep[condition,direction]=pair;logs+=rows;m=B.make_model(cfg,dev);m.load_state_dict(pair[0]);forms.append(dict(seed=seed,direction=direction,condition=condition,**D.aligned_metrics(m,splits[1],splits[2],cfg,dev,direction)));names=list(pair[0]);os=pair[1];pg=os['param_groups'][0]
   for layer,prefix in [('all',''),('encoder','encoder.'),('encoder1','encoder.0'),('encoder2','encoder.2')]:
    states=[os['state'][k] for n,k in zip(names,pg['params']) if n.startswith(prefix)];q=torch.cat([pg['lr']/(torch.sqrt(s['exp_avg_sq'].double().flatten()/(1-.999**float(s['step'])))+pg['eps']) for s in states]);state.append(dict(seed=seed,direction=direction,condition=condition,layer=layer,v_sum=sum(float(s['exp_avg_sq'].double().sum()) for s in states),inverse_scale=float(q.mean())))
  cases=[('acquisition','acquisition'),('acquisition','projected'),('projected','acquisition'),('projected','projected'),('update','update')]
  for recipient,donor in cases:
   ms,own=ep[recipient,direction];other=ep[donor,direction][1];mode='own' if recipient==donor else 'v_other';q=F.q_mean(own);r,n=F.run_branch(cfg,seed,recipient,direction,mode,50,False,ms,own,other,q,0.,splits)
   for row in r+n:row['recipient']=recipient;row['donor']=donor
   traces+=r;norms+=n
 torch.save(ep,folder/'formation_checkpoints.pt')
 for name,rows in [('formation_trace',logs),('formation_endpoints',forms),('state_summary',state),('timecourse',traces),('update_diagnostics',norms)]:F.write_csv(folder/(name+'.csv'),rows)
 dump(folder/'run.json',dict(seed=seed,complete=True,seconds=time.time()-started,source_sha256=sha(Path(root)/'experiments/iv-followup/run_followup.py')));return dict(seed=seed,seconds=round(time.time()-started,2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive-root',type=Path,required=True);ap.add_argument('--phase',choices=['lock','smoke','confirm'],required=True);ap.add_argument('--out',type=Path);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();files=['protocol.md','run_intervention.py','analyze_intervention.py']
 if a.phase=='lock':
  dest=HERE/'registration.json';assert not dest.exists();dump(dest,dict(utc=datetime.now(timezone.utc).isoformat(),seeds=SEEDS,files={f:sha(HERE/f) for f in files},parent_sha256={str(f.relative_to(a.archive_root)):sha(f) for f in [a.archive_root/'experiments/iv-followup/run_followup.py']+list((a.archive_root/'experiments/iv-followup/vendor').rglob('*.py'))},python=platform.python_version(),numpy=np.__version__,torch=torch.__version__));return
 reg=json.loads((HERE/'registration.json').read_text());assert all(sha(HERE/f)==h for f,h in reg['files'].items());assert all(sha(a.archive_root/f)==h for f,h in reg['parent_sha256'].items());a.out.mkdir(exist_ok=False);seeds=[37999] if a.phase=='smoke' else SEEDS;start=time.time()
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for result in pool.map(run_one,[(str(a.archive_root),str(a.out),seed) for seed in seeds]):print(json.dumps(result),flush=True)
 dump(a.out/'provenance.json',dict(seeds=seeds,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',threads_per_worker=1,seconds=time.time()-start,registration_sha256=sha(HERE/'registration.json')))
if __name__=='__main__':main()
