from pathlib import Path
import argparse,json,hashlib
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--archive-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
p=a.archive_root/'experiments/v-a/results_confirm/cell_summary.csv';d=pd.read_csv(p);values={}
for cap in [2,3]:
 x=d[(d.bottleneck==cap)&(d['update']=='balanced')&(d.acquisition!='balanced')]
 y=x.apply(lambda r:r['final_low_q_'+r.acquisition[0]],axis=1)
 assert len(x)==120
 values[str(cap)]={'mean':float(y.mean()),'rows':len(x),'seeds':int(x.seed.nunique())}
out={'description':'Designated-target low-precision allocation score, balanced updating and each preferential acquisition; equal average across 3 targets and 40 seeds. Point summary only.','source':'experiments/v-a/results_confirm/cell_summary.csv','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'by_capacity':values}
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(values))
