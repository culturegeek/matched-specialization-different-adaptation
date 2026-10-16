"""Exploratory replay of unchanged formation with gradient/v accounting."""
import argparse, copy, hashlib, importlib.util, json, platform, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
FIELDS=['seed','direction','route','step','layer','favored_precise','favored_g2','other_g2','dot','cosine','weighted_self','weighted_cross','task_g2','regularized_g2','regularization','v_before','v_after','v_initial','v_self','v_cross','v_regularization','v_rounding','v_reconstructed','inverse_scale','update_g2','m_g2','decomposition_error','recursion_error']
LAYERS=['encoder1','encoder2','head_a','head_b']; SNAP=[0,1,10,50,100,250,500]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def load_parent(root):
 p=root/'experiments/iv-followup/run_followup.py'
 spec=importlib.util.spec_from_file_location('followup_formation_parent',p);f=importlib.util.module_from_spec(spec);sys.modules[spec.name]=f;spec.loader.exec_module(f);return f

def flat(ts): return torch.cat([t.detach().reshape(-1) for t in ts])
def equal_state(a,b):
 if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict): return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
 return a==b


def state_comparison(actual, expected, tolerance):
 """Strict comparison by default; explicit, recorded tensor-specific bounds."""
 report={'bitwise_equal':equal_state(actual,expected),'within_tolerance':True,'tolerance':tolerance,'max_absolute_difference':{},'worst_bound_fraction':0.0}
 def walk(a,b,path='state'):
  if isinstance(a,torch.Tensor):
   if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype:report['within_tolerance']=False;return
   delta=float((a.double()-b.double()).abs().max());kind='v' if path.endswith('exp_avg_sq') else 'm' if path.endswith('exp_avg') else 'step' if path.endswith('step') else 'weights'
   report['max_absolute_difference'][kind]=max(delta,report['max_absolute_difference'].get(kind,0.0))
   scale=float(b.double().abs().max());bound=(1e-12+tolerance*scale) if kind=='v' and tolerance else tolerance*(1+scale) if kind!='step' else 0.0
   ok=bool(torch.isfinite(a).all()) and delta<=bound;report['within_tolerance'] &= ok
   if bound:report['worst_bound_fraction']=max(report['worst_bound_fraction'],delta/bound)
  elif isinstance(a,dict):
   if not isinstance(b,dict) or a.keys()!=b.keys():report['within_tolerance']=False;return
   for k in a:walk(a[k],b[k],path+'.'+str(k))
  elif isinstance(a,(list,tuple)):
   if not isinstance(b,(list,tuple)) or len(a)!=len(b):report['within_tolerance']=False;return
   for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'.'+str(i))
  else:report['within_tolerance'] &= a==b
 walk(actual,expected)
 return report

def run_seed(root,out,seed,tolerance=0.):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 F=load_parent(root);D,B=F.D,F.B;cfg=D.config_for_stage('confirm');dev=torch.device('cpu')
 splits=B.make_splits(cfg,seed);train=splits[0];nm,no=D.prepare_neutral(cfg,seed,train,dev)
 archived=root/f'experiments/iv-followup/results_confirm/seed_{seed}/formation_checkpoints.pt'
 expected=torch.load(archived,map_location='cpu',weights_only=False) if archived.exists() else {}
 dest=out/f'seed_{seed}';dest.mkdir(exist_ok=False);allrows=[];checks=[]
 for di,direction in enumerate(D.DIRECTIONS):
  for ri,route in enumerate(D.ROUTES):
   model=B.make_model(cfg,dev);model.load_state_dict(nm);opt=B.clone_optimizer(model,cfg,no)
   named=list(model.named_parameters());params=[p for _,p in named]
   sizes=[p.numel() for p in params];edges=np.cumsum([0]+sizes)
   idxlayers=[np.concatenate([np.arange(edges[k],edges[k+1]) for k,(n,p) in enumerate(named) if n.startswith(prefix)]) for prefix in ['encoder.0','encoder.2','head_a','head_b']]
   acquisition,w=D.formation_condition(route,direction,.8);wf=w if direction=='A' else 1-w;wo=1-wf
   rng=np.random.default_rng(600000+seed);schedule=D.balanced_schedule(cfg.formation_steps,seed,610000)
   beta=opt.param_groups[0]['betas'][1];decay=cfg.weight_decay
   initial=flat([opt.state[p]['exp_avg_sq'] for p in params]).double().numpy().copy()
   contrib=np.zeros((4,len(initial)),dtype=np.float64);inherited=initial.copy()
   snapshots={k:[] for k in ['weights','m','v']};indices=[]
   def snapshot():
    snapshots['weights'].append(flat(params).numpy().copy())
    for key,outkey in [('exp_avg','m'),('exp_avg_sq','v')]: snapshots[outkey].append(flat([opt.state[p][key] for p in params]).numpy().copy())
   snapshot()
   for step in range(1,cfg.formation_steps+1):
    idx=rng.integers(0,len(train.a),size=cfg.batch_size);indices.append(idx)
    ti=torch.from_numpy(idx);x=D.acquisition_tensor(train,acquisition,int(schedule[step-1]))[ti]
    opt.zero_grad(set_to_none=True);_,pa,pb=model(x);la=((pa-train.a[ti])**2).mean();lb=((pb-train.b[ti])**2).mean()
    ga=torch.autograd.grad(la,params,retain_graph=True,allow_unused=True);gb=torch.autograd.grad(lb,params,retain_graph=True,allow_unused=True)
    ga=flat([torch.zeros_like(p) if g is None else g for g,p in zip(ga,params)]).double().numpy()
    gb=flat([torch.zeros_like(p) if g is None else g for g,p in zip(gb,params)]).double().numpy()
    gf,go=(ga,gb) if direction=='A' else (gb,ga)
    before=flat(params).clone();theta=before.double().numpy();vb=flat([opt.state[p]['exp_avg_sq'] for p in params]).double().numpy()
    loss=w*la+(1-w)*lb;loss.backward()
    gt=flat([p.grad for p in params]).double().numpy()
    # Adam's coupled weight decay is added in parameter dtype.
    gr=flat([p.grad.add(p,alpha=decay) for p in params]).double().numpy()
    terms=np.stack([wf*wf*gf*gf+wo*wo*go*go,2*wf*wo*gf*go,2*decay*theta*(wf*gf+wo*go)+decay*decay*theta*theta])
    opt.step()
    va=flat([opt.state[p]['exp_avg_sq'] for p in params]).double().numpy()
    delta=flat(params).double().numpy()-theta
    # Residual includes gradient and Adam float32 accumulation rounding.
    residual=va-beta*vb-(1-beta)*terms.sum(axis=0)
    contrib[:3]=beta*contrib[:3]+(1-beta)*terms;contrib[3]=beta*contrib[3]+residual;inherited*=beta
    rec=inherited+contrib.sum(axis=0);err=float(np.max(np.abs(rec-va)))
    if err>2e-10: raise RuntimeError(('accounting mismatch',seed,route,direction,step,err))
    m=flat([opt.state[p]['exp_avg'] for p in params]).double().numpy()
    bc=1-beta**(cfg.initial_steps+step);q=cfg.learning_rate/(np.sqrt(va/bc)+opt.param_groups[0]['eps'])
    precise=direction if route=='acquisition' else ('A' if schedule[step-1]==0 else 'B')
    for li,j in enumerate(idxlayers):
     f2=float(gf[j]@gf[j]);o2=float(go[j]@go[j]);dot=float(gf[j]@go[j]);cos=dot/np.sqrt(f2*o2) if f2*o2>0 else 0.
     row=[seed,di,ri,step,li,int(precise==direction),f2,o2,dot,cos,*[float(v[j].sum()) for v in terms[:2]],float(gt[j]@gt[j]),float(gr[j]@gr[j]),float(terms[2,j].sum()),float(vb[j].sum()),float(va[j].sum()),float(inherited[j].sum()),*[float(v[j].sum()) for v in contrib],float(rec[j].sum()),float(q[j].mean()),float(delta[j]@delta[j]),float(m[j]@m[j]),float(np.max(np.abs(gr[j]**2-terms[:,j].sum(axis=0)))),err]
     assert len(row)==len(FIELDS);allrows.append(row)
    if step in SNAP: snapshot()
   actual=(copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()))
   ref=expected.get((route,direction))
   if ref is None: ref=D.form_state(cfg,seed,route,direction,.8,nm,no,train,dev)
   comparison=state_comparison(actual,ref,tolerance);checks.append({'route':route,'direction':direction,**comparison})
   if not comparison['within_tolerance']: raise RuntimeError(('endpoint mismatch',seed,route,direction,comparison))
   np.savez_compressed(dest/f'snapshots_{route}_{direction}.npz',**{k:np.stack(v) for k,v in snapshots.items()},steps=np.array(SNAP),sample_indices=np.array(indices,dtype=np.int16),schedule=schedule,param_names=np.array([n for n,p in named]),param_shapes=np.array([str(tuple(p.shape)) for p in params]),layer_sizes=np.array([len(j) for j in idxlayers]))
 np.savez_compressed(dest/'records.npz',values=np.array(allrows,dtype=np.float64),fields=np.array(FIELDS))
 dump(dest/'validation.json',{'seed':seed,'rows':len(allrows),'endpoints':checks,'checkpoint_sha256':sha(archived) if archived.exists() else None})
 return {'seed':seed,'rows':len(allrows),'bitwise_equal':all(c['bitwise_equal'] for c in checks),'within_tolerance':all(c['within_tolerance'] for c in checks)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--seeds',type=int,nargs='*');p.add_argument('--tolerance',type=float,default=0.,help='Explicit tensor tolerance; default zero is strict. v uses 1e-12 + tolerance * tensor max; weights/m use tolerance * (1 + tensor max); steps exact.');a=p.parse_args();assert np.isfinite(a.tolerance) and a.tolerance>=0
 lock=json.loads((HERE/'analysis_plan_lock.json').read_text())
 for f,h in lock['files'].items(): assert sha(HERE/f)==h,f
 a.out.mkdir(exist_ok=False);start=time.time();seeds=a.seeds or ([19799] if a.smoke else list(range(20000,20040)))
 results=[]
 for seed in seeds:
  x=run_seed(a.archive_root,a.out,seed,a.tolerance);results.append(x);print(json.dumps({**x,'elapsed_s':round(time.time()-start,1)}),flush=True)
 dump(a.out/'provenance.json',{'seeds':seeds,'rows':sum(x['rows'] for x in results),'elapsed_s':time.time()-start,'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'threads':1,'device':'cpu','plan_sha256':sha(HERE/'analysis_plan_lock.json'),'sources':{str(p.relative_to(a.archive_root)):sha(p) for p in (a.archive_root/'experiments/iv-followup').glob('**/experiment_iv_*.py')},'all_endpoints_bitwise_equal':all(x['bitwise_equal'] for x in results),'all_endpoints_within_tolerance':all(x['within_tolerance'] for x in results),'tolerance':a.tolerance,'reproduction_runner_sha256':sha(__file__)})
if __name__=='__main__': main()
