"""IV-F frozen finite robustness and mechanistic experiments (CPU)."""
from pathlib import Path
import argparse,copy,hashlib,json,sys,time,platform
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
import run_followup as F
B,D,E=F.B,F.D,F.E
DEVICE=torch.device('cpu')
MODES=['own','m_other','v_other','reset','v_equal_scale','v_other_equal_scale']
CONFIGS={
 'base':dict(hidden=32,activation='tanh',rho=.35,beta2=.999,steps=500),
 'wide':dict(hidden=64,activation='tanh',rho=.35,beta2=.999,steps=500),
 'relu':dict(hidden=32,activation='relu',rho=.35,beta2=.999,steps=500),
 'rho0':dict(hidden=32,activation='tanh',rho=0.,beta2=.999,steps=500),
 **{f'beta{b}_steps{s}':dict(hidden=32,activation='tanh',rho=.35,beta2=b,steps=s) for b in [.9,.99,.999] for s in [500,1500] if (b,s)!=(.999,500)}}

def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cfg_for(c):
    cfg=D.config_for_stage('confirm');cfg.hidden_dim=c['hidden'];cfg.correlation=c['rho'];cfg.formation_steps=c['steps'];return cfg

def model_for(cfg,c,ms=None):
    model=B.make_model(cfg,DEVICE)
    if c['activation']=='relu':model.encoder[1]=torch.nn.ReLU();model.encoder[3]=torch.nn.ReLU()
    if ms is not None:model.load_state_dict(ms)
    return model

def optimizer_for(model,cfg,c,state=None):
    opt=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate,betas=(.9,c['beta2']),eps=1e-8,weight_decay=cfg.weight_decay)
    if state is not None:opt.load_state_dict(copy.deepcopy(state))
    assert opt.param_groups[0]['betas'][1]==c['beta2']
    return opt

def form(cfg,c,seed,train):
    D.set_seed(seed);m=model_for(cfg,c);o=optimizer_for(m,cfg,c);rng=np.random.default_rng(100000+seed)
    for _ in range(cfg.initial_steps):
        idx=torch.from_numpy(rng.integers(0,len(train.a),cfg.batch_size));B.train_weighted(m,o,train.x_initial[idx],train.a[idx],train.b[idx],.5)
    nm,no=copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict());endpoints={}
    for route in D.ROUTES:
        for direction in D.DIRECTIONS:
            m=model_for(cfg,c,nm);o=optimizer_for(m,cfg,c,no);acq,w=D.formation_condition(route,direction,.8);rng=np.random.default_rng(600000+seed);schedule=D.balanced_schedule(cfg.formation_steps,seed,610000)
            for step in range(cfg.formation_steps):
                idx=torch.from_numpy(rng.integers(0,len(train.a),cfg.batch_size));x=D.acquisition_tensor(train,acq,int(schedule[step]))[idx];B.train_weighted(m,o,x,train.a[idx],train.b[idx],w)
            endpoints[route,direction]=(copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()))
    return endpoints

def state_for(own,other,mode,q,names):
    if not mode.startswith('layer_'):return F.adjusted_state(own,other,mode,q)
    prefix={'layer_encoder1':['encoder.0'],'layer_encoder2':['encoder.2'],'layer_heads':['head_a','head_b']}[mode]
    out=copy.deepcopy(own)
    for k,name in zip(own['param_groups'][0]['params'],names):
        if any(name.startswith(p) for p in prefix):out['state'][k]['exp_avg_sq']=other['state'][k]['exp_avg_sq'].clone()
        assert torch.equal(out['state'][k]['exp_avg'],own['state'][k]['exp_avg'])
        assert torch.equal(out['state'][k]['step'],own['state'][k]['step'])
    return out

def pulse(cfg,c,seed,direction,endpoints,splits,modes,norm_control=False):
    train,probe,test=splits;names=[n for n,_ in model_for(cfg,c).named_parameters()]
    q=np.sqrt(np.prod([F.q_mean(endpoints[r,direction][1]) for r in D.ROUTES]))
    branches=[];rows=[];norms=[]
    for mode in modes:
        for route in D.ROUTES:
            ms,own=endpoints[route,direction];other=endpoints['update' if route=='acquisition' else 'acquisition',direction][1]
            m=model_for(cfg,c,ms);o=optimizer_for(m,cfg,c,state_for(own,other,mode,q,names))
            assert all(torch.equal(m.state_dict()[k],v) for k,v in ms.items())
            ident={'seed':seed,'direction':direction,'route':route,'mode':('norm_' if norm_control else '')+mode}
            start=D.aligned_metrics(m,probe,test,cfg,DEVICE,direction);rows.append({**ident,'step':0,**start});branches.append((m,o,ident))
    rng=np.random.default_rng(700000+seed);order=np.random.default_rng(710000+seed);schedule=[]
    for _ in range(5):p=[0,1];order.shuffle(p);schedule.extend(p)
    acq,w=E.pulse_condition('combined_switch',direction)
    for step in range(1,11):
        idx=torch.from_numpy(rng.integers(0,len(train.a),cfg.batch_size));x=D.acquisition_tensor(train,acq,schedule[step-1])[idx]
        proposed=[]
        for m,o,ident in branches:
            before=[p.detach().clone() for p in m.parameters()];gn=F.gradient(m,x,train.a[idx],train.b[idx],w,cfg.weight_decay);o.step();delta=[p.detach()-v for p,v in zip(m.parameters(),before)];un=float(torch.sqrt(sum((d.double()**2).sum() for d in delta)));assert np.isfinite(un) and un>0
            proposed.append((before,delta,gn,un))
        target=float(np.exp(np.mean(np.log([x[3] for x in proposed])))) if norm_control else None
        for (m,o,ident),(before,delta,gn,un) in zip(branches,proposed):
            if norm_control:
                with torch.no_grad():
                    for p,b,d in zip(m.parameters(),before,delta):p.copy_(b+d*(target/un))
            actual=float(torch.sqrt(sum(((p.detach()-b).double()**2).sum() for p,b in zip(m.parameters(),before))))
            if norm_control:assert abs(actual/target-1)<1e-4
            norms.append({**ident,'step':step,'gradient_norm':gn,'proposed_norm':un,'actual_norm':actual,'target_norm':target if target is not None else un})
    for m,o,ident in branches:rows.append({**ident,'step':10,**D.aligned_metrics(m,probe,test,cfg,DEVICE,direction)})
    return rows,norms

def task(arg):
    name,seed,out,archive,mechanism=arg;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);started=time.time();c=CONFIGS[name];cfg=cfg_for(c);splits=B.make_splits(cfg,seed)
    target=Path(out)/name/f'seed_{seed}';target.mkdir(parents=True,exist_ok=False)
    try:
        if archive is None:endpoints=form(cfg,c,seed,splits[0])
        else:endpoints=torch.load(Path(archive)/f'experiments/iv-followup/results_confirm/seed_{seed}/formation_checkpoints.pt',map_location='cpu',weights_only=True)
        forms=[]
        for (route,direction),(ms,os) in endpoints.items():
            m=model_for(cfg,c,ms);forms.append({'seed':seed,'route':route,'direction':direction,**D.aligned_metrics(m,splits[1],splits[2],cfg,DEVICE,direction)})
        rows=[];norms=[]
        modes=['own','v_other','layer_encoder1','layer_encoder2','layer_heads'] if mechanism else MODES
        for direction in D.DIRECTIONS:
            r,n=pulse(cfg,c,seed,direction,endpoints,splits,modes);rows+=r;norms+=n
            if mechanism:
                r,n=pulse(cfg,c,seed,direction,endpoints,splits,['own','v_other'],True);rows+=r;norms+=n
        F.write_csv(target/'formation_endpoints.csv',forms);F.write_csv(target/'timecourse.csv',rows);F.write_csv(target/'update_diagnostics.csv',norms)
        if archive is None:torch.save(endpoints,target/'formation_checkpoints.pt')
        dump(target/'run.json',{'config':c,'config_full':asdict(cfg),'seed':seed,'seconds':time.time()-started,'source_checkpoint_sha256':sha(Path(archive)/f'experiments/iv-followup/results_confirm/seed_{seed}/formation_checkpoints.pt') if archive else None,'complete':True})
        return name,seed,round(time.time()-started,2)
    except Exception as e:
        dump(target/'failure.json',{'error':repr(e),'elapsed':time.time()-started});raise

def frozen():return [HERE/'protocol.md',HERE/'run_extension.py',HERE/'analyze_extension.py']+sorted((HERE/'vendor').rglob('*.py'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['lock','smoke','general','beta','mechanism'],required=True);ap.add_argument('--out',type=Path);ap.add_argument('--archive-root',type=Path);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    if a.phase=='lock':
        p=HERE/'registration_experiments.json';assert not p.exists();dump(p,{'utc':datetime.now(timezone.utc).isoformat(),'seeds':list(range(32000,32032)),'configs':CONFIGS,'sha256':{str(p.relative_to(HERE)):sha(p) for p in frozen()},'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__});print(p);return
    if a.phase!='smoke':
        registration=json.loads((HERE/'registration_experiments.json').read_text());assert all(sha(HERE/p)==v for p,v in registration['sha256'].items())
    assert a.out is not None and not a.out.exists();a.out.mkdir(parents=True)
    names=['base','wide','relu','rho0'] if a.phase=='general' else [k for k in CONFIGS if k.startswith('beta')] if a.phase=='beta' else ['base']
    seeds=[31999] if a.phase=='smoke' else range(20000,20040) if a.phase=='mechanism' else range(32000,32032)
    args=[(n,s,str(a.out),str(a.archive_root) if a.phase=='mechanism' else None,a.phase=='mechanism') for n in names for s in seeds]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for n,s,t in pool.map(task,args):print(f'{a.phase} {n} seed={s} seconds={t}',flush=True)
    dump(a.out/'manifest.json',{str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print('complete',a.phase,flush=True)
if __name__=='__main__':main()
