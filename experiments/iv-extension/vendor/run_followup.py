"""IV-E prospective follow-up. Frozen parent code, CPU paired interventions."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor/experiment_iv_complete_v05'
spec = importlib.util.spec_from_file_location('ive_original', VENDOR / 'IV-E/experiment_iv_e.py')
E = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = E
spec.loader.exec_module(E)
D, B = E.IVD, E.IVB
MODES = ['own','m_other','v_other','reset','v_equal_scale','v_other_equal_scale',
         'scalar_v_equal','scalar_v_own','sgd_base','sgd_calibrated']
LONG = ['own','m_other','v_other','reset','sgd_calibrated']
RECORD = set(range(11)) | {20,50,100,200,500,510}
DEVICE = torch.device('cpu')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_csv(path, rows):
    E.write_csv(Path(path), rows)


def q_mean(state, scale=1.):
    group = state['param_groups'][0]
    beta2, eps, lr = group['betas'][1], group['eps'], group['lr']
    q = []
    for s in state['state'].values():
        bc = 1 - beta2 ** E.scalar_step(s['step'])
        q.append(lr/(torch.sqrt(s['exp_avg_sq'].double().reshape(-1)*scale/bc)+eps))
    return float(torch.cat(q).mean())


def adjusted_state(own, other, mode, target_q):
    if mode in ['own','m_other','v_other','reset']:
        return E.compose_optimizer_state(own, other, mode)
    # Only v is transplanted; m/step/groups always come from the recipient.
    state = E.compose_optimizer_state(own, other, 'v_other') if mode == 'v_other_equal_scale' else copy.deepcopy(own)
    if mode.startswith('scalar_v'):
        target = q_mean(own) if mode == 'scalar_v_own' else target_q
        g = state['param_groups'][0]
        for s in state['state'].values():
            bc = 1 - g['betas'][1] ** E.scalar_step(s['step'])
            v = (g['lr']/target-g['eps'])**2*bc
            s['exp_avg_sq'].fill_(v)
    else:
        # Monotone solve including epsilon; no outcome is used.
        lo, hi = -30., 30.
        for _ in range(80):
            mid = (lo+hi)/2
            if q_mean(state, np.exp(mid)) > target_q:
                lo = mid
            else:
                hi = mid
        factor = np.exp((lo+hi)/2)
        for s in state['state'].values():
            s['exp_avg_sq'].mul_(factor)
    for k in own['state']:
        assert torch.equal(state['state'][k]['exp_avg'], own['state'][k]['exp_avg'])
        assert E.scalar_step(state['state'][k]['step']) == E.scalar_step(own['state'][k]['step'])
    expected = q_mean(own) if mode == 'scalar_v_own' else target_q
    assert abs(q_mean(state)/expected-1) < 2e-6
    return state


def gradient(model, x, a, b, w, decay):
    model.zero_grad(set_to_none=True)
    _, pa, pb = model(x)
    loss = w*((pa-a)**2).mean() + (1-w)*((pb-b)**2).mean()
    loss.backward()
    grads = [p.grad.detach()+decay*p.detach() for p in model.parameters()]
    return float(torch.sqrt(sum((g.double()**2).sum() for g in grads)))


def first_update_calibration(cfg, seed, direction, endpoints, train):
    rng = np.random.default_rng(700_000+seed)
    idx = torch.from_numpy(rng.integers(0, len(train.a), size=cfg.batch_size))
    acq, w = E.pulse_condition('combined_switch', direction)
    x = D.acquisition_tensor(train, acq, 0)[idx]
    scales, diagnostics = [], []
    for route in D.ROUTES:
        ms, os = endpoints[(route,direction)]
        model = B.make_model(cfg, DEVICE); model.load_state_dict(ms)
        opt = B.clone_optimizer(model,cfg,os)
        gn = gradient(model,x,train.a[idx],train.b[idx],w,cfg.weight_decay)
        before = [p.detach().clone() for p in model.parameters()]
        opt.step()
        un = float(torch.sqrt(sum(((p.detach()-v).double()**2).sum() for p,v in zip(model.parameters(),before))))
        scales.append(un/gn)
        diagnostics.append({'seed':seed,'direction':direction,'route':route,
                            'first_gradient_norm':gn,'first_adam_update_norm':un,
                            'adam_norm_over_gradient':un/gn})
    lr = float(np.sqrt(scales[0]*scales[1]))
    for r in diagnostics: r['paired_sgd_lr'] = lr
    return lr, diagnostics


def run_branch(cfg, seed, route, direction, mode, horizon, post_neutral,
               ms, own, other, target_q, sgd_lr, splits):
    train,probe,test = splits
    model = B.make_model(cfg, DEVICE); model.load_state_dict(ms)
    if mode.startswith('sgd'):
        opt = torch.optim.SGD(model.parameters(),lr=sgd_lr if mode=='sgd_calibrated' else cfg.learning_rate,
                              momentum=0,weight_decay=cfg.weight_decay)
        assert not opt.state
    else:
        opt = torch.optim.Adam(model.parameters(),lr=cfg.learning_rate,betas=(.9,.999),eps=1e-8,
                               weight_decay=cfg.weight_decay)
        state = adjusted_state(own,other,mode,target_q)
        if state is not None: opt.load_state_dict(state)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in ms.items())
    rng = np.random.default_rng(700_000+seed)
    order_rng = np.random.default_rng(710_000+seed)
    schedule = []
    for _ in range((horizon+1)//2):
        pair = [0,1]; order_rng.shuffle(pair); schedule.extend(pair)
    rows, norms = [], []
    identity = {'seed':seed,'route':route,'direction':direction,'mode':mode,
                'trajectory':'pulse_then_neutral' if post_neutral else 'continued_opposition'}
    def record(step):
        rows.append({**identity,'step':step,**D.aligned_metrics(model,probe,test,cfg,DEVICE,direction)})
    record(0)
    for step in range(1,horizon+1):
        idx = torch.from_numpy(rng.integers(0,len(train.a),size=cfg.batch_size))
        if post_neutral and step>10: acq,w='balanced',.5
        else: acq,w=E.pulse_condition('combined_switch',direction)
        x = D.acquisition_tensor(train,acq,schedule[step-1])[idx]
        gn = gradient(model,x,train.a[idx],train.b[idx],w,cfg.weight_decay)
        before = [p.detach().clone() for p in model.parameters()]
        opt.step()
        un = float(torch.sqrt(sum(((p.detach()-v).double()**2).sum() for p,v in zip(model.parameters(),before))))
        norms.append({**identity,'step':step,'gradient_norm':gn,'update_norm':un,
                      'm_norm':float(np.sqrt(sum(float((s['exp_avg'].double()**2).sum()) for s in opt.state.values() if 'exp_avg' in s))),
                      'v_sum':sum(float(s['exp_avg_sq'].double().sum()) for s in opt.state.values() if 'exp_avg_sq' in s)})
        if step in RECORD: record(step)
        if not np.isfinite(un): raise RuntimeError(f'nonfinite update {identity} step={step}')
    return rows,norms


def frozen_files():
    return [Path(__file__),HERE/'protocol.md',HERE/'analyze_followup.py',
            HERE/'existing_analysis/frozen_models.json'] + [VENDOR/f'IV-{x}/experiment_iv_{x.lower()}.py' for x in ['B','D','E']]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage',choices=['smoke','confirm','lock','reproduce'],required=True)
    p.add_argument('--out',default=None)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.stage=='lock':
        path=HERE/'registration.json'
        if path.exists(): raise RuntimeError('Registration already exists')
        dump(path,{'time_utc':datetime.now(timezone.utc).isoformat(),
                   'seeds':list(range(20000,20040)),'modes':MODES,
                   'file_hashes':{str(f.relative_to(HERE)):sha(f) for f in frozen_files()}})
        print(path); return
    seeds = [19799] if args.stage=='smoke' else [9100] if args.stage=='reproduce' else list(range(20000,20040))
    if args.stage=='confirm':
        reg=json.loads((HERE/'registration.json').read_text())
        assert reg['seeds']==seeds and reg['modes']==MODES
        for f,h in reg['file_hashes'].items(): assert sha(HERE/f)==h,f
    cfg=D.config_for_stage('confirm')
    out=HERE/(args.out or f'results_{args.stage}')
    out.mkdir(exist_ok=True)
    if (out/'provenance.json').exists(): raise RuntimeError('Refuse to overwrite completed run')
    forms,moments,calibrations=[],[],[]
    started=time.time()
    for seed in seeds:
        target=out/f'seed_{seed}'
        if target.exists(): raise RuntimeError(f'Refuse to overwrite {target}')
        target.mkdir()
        splits=B.make_splits(cfg,seed); train,probe,test=splits
        nm,no=D.prepare_neutral(cfg,seed,train,DEVICE)
        endpoints={}
        for direction in D.DIRECTIONS:
            for route in D.ROUTES:
                ms,os=D.form_state(cfg,seed,route,direction,.8,nm,no,train,DEVICE)
                assert os['param_groups'][0]['betas']==(.9,.999)
                assert os['param_groups'][0]['eps']==1e-8
                assert all(E.scalar_step(s['step'])==600 for s in os['state'].values())
                endpoints[(route,direction)]=(ms,os)
                model=B.make_model(cfg,DEVICE);model.load_state_dict(ms)
                forms.append({'seed':seed,'route':route,'direction':direction,
                              **D.aligned_metrics(model,probe,test,cfg,DEVICE,direction)})
                layers=E.optimizer_moment_rows(seed,route,direction,ms,os,cfg,DEVICE)
                moments.append({'seed':seed,'route':route,'direction':direction,
                                **E.aggregate_moment_endpoint(layers)})
        torch.save(endpoints,target/'formation_checkpoints.pt')
        rows,norms=[],[]
        for direction in D.DIRECTIONS:
            lr,cal=first_update_calibration(cfg,seed,direction,endpoints,train);calibrations.extend(cal)
            common_q=np.sqrt(q_mean(endpoints[('acquisition',direction)][1])*q_mean(endpoints[('update',direction)][1]))
            for route in D.ROUTES:
                ms,own=endpoints[(route,direction)];other=endpoints[('update' if route=='acquisition' else 'acquisition',direction)][1]
                modes=['own','m_other','v_other','reset'] if args.stage=='reproduce' else MODES
                for mode in modes:
                    horizon=500 if mode in LONG and args.stage!='reproduce' else 10
                    r,n=run_branch(cfg,seed,route,direction,mode,horizon,False,ms,own,other,common_q,lr,splits)
                    rows.extend(r);norms.extend(n)
                if args.stage!='reproduce':
                    for mode in ['own','reset']:
                        r,n=run_branch(cfg,seed,route,direction,mode,510,True,ms,own,other,common_q,lr,splits)
                        rows.extend(r);norms.extend(n)
        write_csv(target/'timecourse.csv',rows)
        write_csv(target/'update_diagnostics.csv',norms)
        print(f'{args.stage}: seed {seed} complete; elapsed {time.time()-started:.1f}s',flush=True)
    write_csv(out/'formation_endpoints.csv',forms)
    write_csv(out/'optimizer_moments.csv',moments)
    write_csv(out/'sgd_calibration.csv',calibrations)
    dump(out/'provenance.json',{'stage':args.stage,'seeds':seeds,'config':asdict(cfg),
                              'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                              'threads':1,'deterministic':True,'device':'cpu','elapsed_seconds':time.time()-started,
                              'file_hashes':{str(f.relative_to(HERE)):sha(f) for f in frozen_files()},
                              'registration_sha256':sha(HERE/'registration.json') if args.stage=='confirm' else None})


if __name__=='__main__':main()
