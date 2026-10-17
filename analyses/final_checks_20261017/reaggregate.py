"""Independent csv-based verification and fixed-cohort gradient summaries."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();root=a.archive_root;study=root/'experiments/iv-cancellation';a.out.mkdir(exist_ok=False)
 seeds=list(range(38000,38040));horizons=[1,5,10,20,50];pairs=[('acquisition','acquisition'),('acquisition','projected'),('projected','acquisition'),('projected','projected'),('update','update')];raw={t:{p:[] for p in pairs} for t in horizons};trace={};inputs={};max_identity=0.0
 def rows(f):
  inputs[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
  with f.open() as h:return list(csv.DictReader(h))
 for seed in seeds:
  folder=study/'runs'/f'seed_{seed}';tc=rows(folder/'timecourse.csv');val={(r['direction'],r['recipient'],r['donor'],int(r['step'])):float(r['aligned_s']) for r in tc};assert len(val)==130
  for t in horizons:
   for rc,do in pairs:raw[t][rc,do].append(np.mean([val[dr,rc,do,0]-val[dr,rc,do,t] for dr in ['A','B']]))
  tr=rows(folder/'formation_trace.csv');assert len(tr)==3000
  for cond in ['acquisition','projected','update']:
   for window,low,high in [('all',1,500),('early',1,250),('late',251,500),('v_weighted',1,500)]:
    sums={f:0. for f in ['pre_self','pre_cross','post_self','post_cross','pre_norm2','post_norm2']};last=[]
    for dr in ['A','B']:
     rr=sorted([r for r in tr if r['condition']==cond and r['direction']==dr],key=lambda r:int(r['step']));assert [int(r['step']) for r in rr]==list(range(1,501));last.append(float(rr[-1]['v_encoder']))
     for r in rr:
      for prefix in ['pre','post']:
       err=abs(float(r[prefix+'_norm2'])-float(r[prefix+'_self'])-float(r[prefix+'_cross']));max_identity=max(max_identity,err);assert err<1e-12+1e-10*float(r[prefix+'_self'])
      t=int(r['step'])
      if not low<=t<=high:continue
      w=.001*.999**(500-t) if window=='v_weighted' else 1/(high-low+1)
      for f in sums:sums[f]+=w*float(r[f])/2
    sums.update(cancellation_before=-sums['pre_cross']/sums['pre_self'],cancellation_after=-sums['post_cross']/sums['post_self'],individual_square_post_over_pre=sums['post_self']/sums['pre_self'],combined_square_post_over_pre=sums['post_norm2']/sums['pre_norm2'],final_v=np.exp(np.log(last).mean()))
    trace.setdefault((window,cond),[]).append(sums)
 idx=np.random.default_rng(2026101704).integers(40,size=(10000,40))
 def ci(x,log=False):
  z=np.asarray(x);z=np.log(z) if log else z;f=np.exp if log else lambda x:x;b=z[idx].mean(1);return dict(mean=float(f(z.mean())),low=float(f(np.quantile(b,.025))),high=float(f(np.quantile(b,.975))),n=40)
 response={};source=json.loads((study/'results/result.json').read_text());checks=0;maxdiff=0.
 for t,v in raw.items():
  v={k:np.array(x) for k,x in v.items()};b=v['acquisition','acquisition'];pr=v['projected','projected'];anchor=v['update','update'];e={'response_'+rc+'_'+do:ci(x) for (rc,do),x in v.items()}
  for name,x in [('native_route_contrast',b-anchor),('projected_route_contrast',pr-anchor),('formation_total_effect',pr-b),('v_donor_effect_native_recipient',v['acquisition','projected']-b),('v_donor_effect_projected_recipient',pr-v['projected','acquisition'])]:e[name]=ci(x)
  for key,x in e.items():
   for field in ['mean','low','high']:
    diff=abs(x[field]-source['response'][str(t)][key][field]);assert diff<1e-12;(checks:=checks+1);maxdiff=max(maxdiff,diff)
  response[str(t)]=e
 summary={}
 for (window,cond),values in trace.items():
  summary.setdefault(window,{})[cond]={k:ci([v[k] for v in values],k.endswith('_over_pre') or k=='final_v') for k in values[0]}
 ratios={}
 for window in ['all','early','late','v_weighted']:
  ratios[window]={}
  for field in ['pre_self','post_self','pre_norm2','post_norm2','final_v']:
   for c in ['projected','update']:
    z=[x[field]/y[field] for x,y in zip(trace[window,c],trace[window,'acquisition'])];ratios[window][c+'_over_acquisition_'+field]=ci(z,True)
 result=dict(status='PASS',scope='Independent verification of known response estimates; exploratory saved-trace decomposition, no new training.',seeds=seeds,horizons=horizons,response_estimates_verified=checks,max_response_difference=maxdiff,max_decomposition_identity_error=max_identity,response=response,trace=summary,ratios=ratios,input_sha256=inputs)
 (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['status','response_estimates_verified','max_response_difference','max_decomposition_identity_error']},indent=2));print(json.dumps(summary['all']['projected'],indent=2));print(json.dumps(ratios['all'],indent=2))
if __name__=='__main__':main()
