"""Exploratory endpoint-response and layer analyses; no retraining."""
from pathlib import Path
import argparse,json,hashlib,sys,importlib.util
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
import run_followup as F

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interval(x,rng):
    x=np.asarray(x);samples=x[rng.integers(0,len(x),(5000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'n':len(x)}
def fit(X,y):
    mu=X.mean(0);sd=X.std(0);sd=np.where(sd>1e-12,sd,1)
    c=np.linalg.lstsq(np.c_[np.ones(len(X)),(X-mu)/sd],y,rcond=None)[0]
    slope=c[1:]/sd
    return np.r_[c[0]-mu@slope,slope]
def predict(c,X):return np.c_[np.ones(len(X)),X]@c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    rng=np.random.default_rng(2026101604);cohorts={};inputhash={};paired=[]
    metrics=['aligned_s','target_D','other_D','error_sum','aligned_low_s'];features=['target_D','other_D','aligned_low_s']
    for cohort,sub in [('IV-D','iv/IV-D'),('IV-E','iv/IV-E'),('followup','iv-followup')]:
        root=a.archive_root/'experiments'/sub/'results_confirm'
        p=root/'formation_endpoints.csv';forms=pd.read_csv(p);inputhash[str(p.relative_to(a.archive_root))]=sha(p)
        diff=forms.groupby(['seed','route'])[metrics].mean().unstack('route')
        diff=pd.DataFrame({m:diff[m]['acquisition']-diff[m]['update'] for m in metrics})
        if cohort=='followup':
            p=root/'seed_route_effects.csv';raw=pd.read_csv(p);y=raw.query("mode=='own' and trajectory=='continued_opposition' and step==10").set_index('seed').route_drop_difference
        else:
            p=root/'protocol_summary.csv';raw=pd.read_csv(p);col,mode=('optimizer_mode','carried') if cohort=='IV-D' else ('optimizer_component','own')
            raw=raw[(raw[col]==mode)&(raw.protocol=='combined_switch')]
            r=raw.groupby(['seed','route']).pulse_drop.mean().unstack('route');y=r.acquisition-r['update']
        inputhash[str(p.relative_to(a.archive_root))]=sha(p)
        diff['response']=y;assert len(diff)==40 and not diff.isna().any().any()
        diff['cohort']=cohort;paired.append(diff.reset_index());cohorts[cohort]=diff
    summary={'status':'exploratory_existing_data','cohorts':{},'layer_summary':[]}
    predictions=[]
    for cohort,diff in cohorts.items():
        X=diff[features].to_numpy();y=diff.response.to_numpy();n=len(y);cor={}
        for m in metrics:
            x=diff[m].to_numpy();idx=rng.integers(0,n,(5000,n));xx=x[idx];yy=y[idx];xx-=xx.mean(1,keepdims=True);yy-=yy.mean(1,keepdims=True)
            bc=(xx*yy).sum(1)/np.sqrt((xx*xx).sum(1)*(yy*yy).sum(1));cor[m]={'r':float(np.corrcoef(x,y)[0,1]),'ci95':np.quantile(bc,[.025,.975]).tolist()}
        pred=[];base=[]
        for j in range(n):
            keep=np.arange(n)!=j;pred.append(predict(fit(X[keep],y[keep]),X[j:j+1])[0]);base.append(y[keep].mean())
            predictions.append({'cohort':cohort,'seed':int(diff.index[j]),'observed':y[j],'predicted':pred[-1],'baseline':base[-1]})
        ratio=float(np.mean((np.array(pred)-y)**2)/np.mean((np.array(base)-y)**2));c=fit(X,y)
        bi=[]
        for _ in range(5000):
            ix=rng.integers(0,n,n);bi.append(fit(X[ix],y[ix])[0])
        summary['cohorts'][cohort]={'response':interval(y,rng),'correlations':cor,'loo_mse_ratio':ratio,'zero_endpoint_difference_prediction':{'mean':float(c[0]),'ci95':np.quantile(bi,[.025,.975]).tolist()},'coefficients':c.tolist(),'features':features}
    c=fit(cohorts['IV-D'][features].to_numpy(),cohorts['IV-D'].response.to_numpy());base=cohorts['IV-D'].response.mean()
    summary['transfer_from_IV_D']={}
    for cohort in ['IV-E','followup']:
        df=cohorts[cohort];y=df.response.to_numpy();pred=predict(c,df[features].to_numpy())
        summary['transfer_from_IV_D'][cohort]={'mse_ratio_to_training_mean':float(np.mean((pred-y)**2)/np.mean((base-y)**2))}
    layers=[];cfg=F.D.config_for_stage('confirm');names=[k for k,_ in F.B.make_model(cfg,F.DEVICE).named_parameters()]
    for seed in range(20000,20040):
        p=a.archive_root/f'experiments/iv-followup/results_confirm/seed_{seed}/formation_checkpoints.pt';inputhash[str(p.relative_to(a.archive_root))]=sha(p)
        endpoints=torch.load(p,map_location='cpu',weights_only=True)
        for (route,direction),(ms,os) in endpoints.items():
            g=os['param_groups'][0];by={}
            for name,k in zip(names,g['params']):
                layer='encoder1' if name.startswith('encoder.0') else 'encoder2' if name.startswith('encoder.2') else 'target_head' if name.startswith('head_'+direction.lower()) else 'other_head'
                state=os['state'][k];v=state['exp_avg_sq'].double().flatten();bc=1-g['betas'][1]**float(state['step']);q=g['lr']/(torch.sqrt(v/bc)+g['eps']);by.setdefault(layer,[]).append((v,q,bc))
            total=sum(float(v.sum()) for vals in by.values() for v,q,bc in vals)
            for layer,vals in by.items():
                v=torch.cat([v for v,q,bc in vals]);q=torch.cat([q for v,q,bc in vals]);layers.append({'seed':seed,'route':route,'direction':direction,'layer':layer,'n_parameters':v.numel(),'v_sum':float(v.sum()),'v_share':float(v.sum())/total,'v_per_parameter':float(v.mean()),'vhat_per_parameter':float(v.mean()/vals[0][2]),'q_mean':float(q.mean())})
    ld=pd.DataFrame(layers);ld.to_csv(a.out/'layer_values.csv',index=False)
    for (route,layer),df in ld.groupby(['route','layer']):
        grouped=df.groupby('seed').mean(numeric_only=True)
        summary['layer_summary'].append({'route':route,'layer':layer,'n_parameters':int(df.n_parameters.iloc[0]),**{m:interval(grouped[m],rng) for m in ['v_share','v_per_parameter','vhat_per_parameter','q_mean']}})
    pd.concat(paired).to_csv(a.out/'paired_endpoint_response.csv',index=False);pd.DataFrame(predictions).to_csv(a.out/'loo_predictions.csv',index=False)
    (a.out/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');(a.out/'inputs.json').write_text(json.dumps(inputhash,indent=2)+'\n')
    print(json.dumps({k:{x:v[x] for x in ['response','loo_mse_ratio','zero_endpoint_difference_prediction']} for k,v in summary['cohorts'].items()},indent=2))
if __name__=='__main__':main()
