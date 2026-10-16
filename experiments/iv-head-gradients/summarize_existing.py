from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--archive-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();P=a.archive_root;res=a.out;res.mkdir(parents=True,exist_ok=False)

frames=[]
for sub in ['results_beta','results_general']:
 df=pd.read_csv(P/f'experiments/iv-extension/{sub}/seed_effects.csv');df=df if sub=='results_beta' else df[df.config=='base'].assign(config='beta0.999_steps500');frames.append(df)
f=pd.concat(frames);f['reset_minus_own']=f.reset-f.own;f[['config','seed','reset_minus_own']].to_csv(res/'reset_seed_differences.csv',index=False);out={};idx=np.random.default_rng(2026101611).integers(0,32,(5000,32))
for name,d in f.groupby('config'):
 x=d.sort_values('seed').reset_minus_own.to_numpy();lo,hi=np.quantile(x[idx].mean(1),[.025,.975]);out[name]={'mean':float(x.mean()),'low':float(lo),'high':float(hi),'n':32}
(res/'reset_comparison.json').write_text(json.dumps({'status':'exploratory paired comparison; existing criteria unchanged','by_setting':out},indent=2)+'\n')
print('Reset increases', {k:v for k,v in out.items() if k.startswith('beta0.9_')})
# A claim inventory derived from original frozen result JSONs.
rows=[]
for sub in ['results_general','results_beta']:
 ds=json.loads((P/f'experiments/iv-extension/{sub}/result.json').read_text())
 for k,d in ds.items():
  e=d['effects'];v=e['v_other']['ci95'];o=e['own']['ci95'];rs=e['reset']['ci95'];pat=d['contrasts']['scale_controlled_pattern_effect']['ci95'];
  rows.append({'setting':k,'study':'IV-F','source':f'experiments/iv-extension/{sub}/result.json','endpoint_match':'Y' if d['gates']['endpoint_match_all'] else 'N','pattern_effect':'Y' if pat[0]>0 else 'N' if pat[1]<0 else '?','reversal':'Y' if o[0]>0 and v[1]<0 else 'N' if v[0]>0 else '?','reset_equivalent':'Y' if rs[0]>-.02 and rs[1]<.02 else 'N' if rs[0]>.02 or rs[1]<-.02 else '?'})
for study,folder in [('V-B','v-three-adaptation'),('V-C','v-natural')]:
 ds=json.loads((P/f'experiments/{folder}/results_confirm/result.json').read_text())
 for cap,d in ds['by_capacity'].items():
  e=d['horizons']['10']['effects'];c=d['horizons']['10']['contrasts'];rs=e['reset'];v=e['v_other'];o=e['own'];rows.append({'setting':f'{study}_capacity{cap}','study':study,'source':f'experiments/{folder}/results_confirm/result.json','endpoint_match':'Y' if d['all_endpoint_gates_pass'] else 'N','pattern_effect':'Y' if c['scale_controlled_pattern_effect']['low']>0 else '?','reversal':'Y' if o['low']>0 and v['high']<0 else 'N' if v['low']>0 else '?','reset_equivalent':'Y' if rs['low']>-.02 and rs['high']<.02 else 'N' if rs['low']>.02 or rs['high']<-.02 else '?'})
assert len(rows)==13;pd.DataFrame(rows).to_csv(res/'claim_inventory.csv',index=False)
