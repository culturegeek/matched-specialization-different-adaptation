from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import argparse
p=argparse.ArgumentParser();p.add_argument('--archive-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();P=a.archive_root;S=a.out;S.mkdir(parents=True,exist_ok=False)
paths=['experiments/iv-head-gradients/results/batch_gradients.csv','experiments/iv-extension/results_mechanism/seed_effects.csv','experiments/iv-head-gradients/existing_results/claim_inventory.csv']
b=pd.read_csv(P/paths[0]);assert len(b)==40960
# Independently reconstruct the own-state response contrast from all saved trajectories.
responses=[]
for seed in range(20000,20040):
 path=f'experiments/iv-extension/runs_mechanism/base/seed_{seed}/timecourse.csv';paths.append(path);t=pd.read_csv(P/path);t=t[t['mode']=='own'];q=t.pivot(index=['direction','route'],columns='step',values='aligned_s');drops=(q[0]-q[10]).unstack('route');responses.append({'seed':seed,'response_contrast':float((drops.acquisition-drops['update']).mean())})
y=pd.DataFrame(responses).set_index('seed').response_contrast.sort_index();old=pd.read_csv(P/'experiments/iv-extension/results_mechanism/seed_effects.csv').set_index('seed').own;assert np.allclose(y,old,atol=1e-15,rtol=0)
# Exploratory endpoint association after prior results were known.
# Match the archived convention: batch means per checkpoint, log ratios by direction,
# direction average within seed; correlations are across 40 paired seeds.
c=b.groupby(['input','layer','seed','direction','route']).mean(numeric_only=True)
idx=np.random.default_rng(2026101701).integers(0,40,(20000,40))
def pearson(x,z):
 x=np.asarray(x);z=np.asarray(z);xx=x-x.mean(-1,keepdims=True);zz=z-z.mean(-1,keepdims=True);return (xx*zz).sum(-1)/np.sqrt((xx*xx).sum(-1)*(zz*zz).sum(-1))
records=[];seedrows=[];descriptive={}
for inp in ['both_high','both_low','favored_precise','other_precise']:
 descriptive[inp]={}
 for layer in ['encoder1','encoder2']:
  d=c.xs((inp,layer),level=['input','layer']);p=d.unstack('route');lr=(.5*np.log(p.equal_total_g2.acquisition/p.equal_total_g2['update'])).groupby('seed').mean().sort_index();vars={'acquisition_cosine':d.xs('acquisition',level='route').cosine.groupby('seed').mean().sort_index(),'update_cosine':d.xs('update',level='route').cosine.groupby('seed').mean().sort_index(),'log_equal_total_rms_ratio':lr}
  descriptive[inp][layer]={}
  for route in ['acquisition','update']:
   rr=d.xs(route,level='route');f2=float(rr.favored_g2.mean());o2=float(rr.other_g2.mean());descriptive[inp][layer][route]={'mean_favored_squared_norm':f2,'mean_other_squared_norm':o2,'pooled_favored_rms':float(np.sqrt(f2)),'pooled_other_rms':float(np.sqrt(o2)),'other_to_favored_squared_ratio':o2/f2,'other_to_favored_pooled_rms_ratio':float(np.sqrt(o2/f2)),'mean_cosine':float(rr.cosine.mean()),'equal_total_mean_squared':float(rr.equal_total_g2.mean()),'formation_total_mean_squared':float(rr.formation_total_g2.mean())}
  for name,x in vars.items():
   assert np.array_equal(x.index,y.index);xx=x.to_numpy();yy=y.to_numpy();boots=pearson(xx[idx],yy[idx]);rec={'input':inp,'layer':layer,'predictor':name,'pearson':float(pearson(xx,yy)),'ci95_percentile':np.quantile(boots,[.025,.975]).tolist(),'n':40,'predictor_mean':float(xx.mean()),'predictor_sd':float(xx.std(ddof=1))};records.append(rec)
   for seed,xv,yv in zip(x.index,xx,yy):seedrows.append({'input':inp,'layer':layer,'predictor':name,'seed':int(seed),'x':xv,'response_contrast':yv})
inv=pd.read_csv(P/paths[2]);counts={name:inv[name].value_counts().to_dict() for name in ['endpoint_match','pattern_effect','reversal','reset_equivalent']}
out={'status':'exploratory endpoint association; not prespecified confirmatory inference','correlation_definition':'Pearson across seeds after within-seed averaging of the two target directions; 20000 paired seed bootstrap resamples, RNG 2026101701; individual intervals, no multiplicity adjustment. All 4 probes and 2 layers retained.','source_hashes':{p:hashlib.sha256((P/p).read_bytes()).hexdigest() for p in paths},'claims':counts,'descriptive':descriptive,'correlations':records}
(S/'result.json').write_text(json.dumps(out,indent=2)+'\n');pd.DataFrame(seedrows).to_csv(S/'seed_correlations.csv',index=False)
