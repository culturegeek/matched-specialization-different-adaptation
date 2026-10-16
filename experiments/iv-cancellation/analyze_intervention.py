"""Fixed paired estimates for the locally specified PCGrad study."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();folders=sorted(a.input.glob('seed_*'));seeds=[int(p.name.split('_')[-1]) for p in folders];assert seeds==list(range(38000,38040));a.out.mkdir(exist_ok=False)
 def read(name):return pd.concat([pd.read_csv(p/(name+'.csv')) for p in folders],ignore_index=True)
 rng=np.random.default_rng(2026101704);boot=rng.integers(40,size=(10000,40))
 def ci(x,log=False):
  x=np.asarray(x,float);assert x.shape==(40,);z=np.log(x) if log else x;b=z[boot].mean(1);fn=np.exp if log else lambda x:x
  return dict(mean=float(fn(z.mean())),low=float(fn(np.quantile(b,.025))),high=float(fn(np.quantile(b,.975))),n=40)
 result=dict(status='locally specified prospective intervention following exploratory diagnostics',seeds=seeds,bootstrap_seed=2026101704,bootstrap_samples=10000,state={},formation_match={},response={},trace={})
 state=read('state_summary');state.to_csv(a.out/'state_summary.csv',index=False)
 for layer,g in state.groupby('layer'):
  result['state'][layer]={}
  for k in ['v_sum','inverse_scale']:
   w=g.pivot(index=['seed','direction'],columns='condition',values=k);ratio=np.exp(np.log(w.projected/w.acquisition).groupby('seed').mean());result['state'][layer][k+'_projected_over_native']=ci(ratio,True)
 forms=read('formation_endpoints');forms.to_csv(a.out/'formation_endpoints.csv',index=False)
 for condition in ['acquisition','projected']:
  result['formation_match'][condition]={}
  for k in ['aligned_s','target_D','other_D','error_sum','aligned_low_s']:
   w=forms.pivot(index=['seed','direction'],columns='condition',values=k);q=ci((w[condition]-w['update']).groupby('seed').mean());q['equivalent']=q['low']>=-.02 and q['high']<=.02;result['formation_match'][condition][k]=q
 raw=read('timecourse');keys=['seed','direction','recipient','donor'];start=raw[raw.step==0].set_index(keys).aligned_s;response=[]
 for t in [1,5,10,20,50]:
  end=raw[raw.step==t].set_index(keys).aligned_s;v=(start-end).rename('response').reset_index();v['step']=t;response.append(v)
 df=pd.concat(response);df.to_csv(a.out/'responses.csv',index=False)
 for t,g in df.groupby('step'):
  w=g.pivot(index=['seed','direction'],columns=['recipient','donor'],values='response').groupby('seed').mean();base=w['acquisition','acquisition'];projected=w['projected','projected'];anchor=w['update','update'];entry={}
  for key in w.columns:entry['response_'+key[0]+'_'+key[1]]=ci(w[key])
  for name,v in [('native_route_contrast',base-anchor),('projected_route_contrast',projected-anchor),('formation_total_effect',projected-base),('v_donor_effect_native_recipient',w['acquisition','projected']-base),('v_donor_effect_projected_recipient',projected-w['projected','acquisition'])]:entry[name]=ci(v)
  result['response'][str(t)]=entry
 logs=read('formation_trace');summ=logs.groupby(['seed','condition'])[['pre_self','pre_cross','post_self','post_cross','pre_norm2','post_norm2','conflict']].mean().reset_index();summ['cancellation_before']=-summ.pre_cross/summ.pre_self;summ['cancellation_after']=-summ.post_cross/summ.post_self;summ.to_csv(a.out/'trace_seed_summary.csv',index=False)
 for cond,g in summ.groupby('condition'):result['trace'][cond]={k:ci(g.sort_values('seed')[k]) for k in ['cancellation_before','cancellation_after','pre_norm2','post_norm2','conflict']}
 result['input_sha256']={str(f.relative_to(a.input)):hashlib.sha256(f.read_bytes()).hexdigest() for p in folders for f in sorted(p.glob('*.csv'))};(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['state','formation_match','response']},indent=2))
if __name__=='__main__':main()
