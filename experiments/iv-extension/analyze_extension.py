"""Pre-fixed paired analyses of the IV-F finite experimental set."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd

def interval(v,rng):
    v=np.asarray(v);assert np.isfinite(v).all();means=v[rng.integers(0,len(v),(5000,len(v)))].mean(1)
    return {'mean':float(v.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'n':len(v)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(2026101605)
    records={};allseeds=[];inputs={};normrecords=[]
    for folder in sorted(p for p in a.input.iterdir() if p.is_dir()):
        t=[];f=[];n=[]
        for run in sorted(folder.glob('seed_*')):
            assert json.loads((run/'run.json').read_text())['complete']
            for name,ls in [('timecourse.csv',t),('formation_endpoints.csv',f),('update_diagnostics.csv',n)]:
                p=run/name;ls.append(pd.read_csv(p));inputs[str(p.relative_to(a.input))]=hashlib.sha256(p.read_bytes()).hexdigest()
        tc=pd.concat(t);fc=pd.concat(f);nc=pd.concat(n);assert tc.seed.nunique() in [1,32,40]
        p=tc.pivot(index=['seed','direction','route','mode'],columns='step',values='aligned_s');p['drop']=p[0]-p[10]
        means=p['drop'].groupby(['seed','route','mode']).mean().unstack('route');d=(means.acquisition-means['update']).unstack('mode');d.columns.name=None
        ds=d.reset_index();ds.insert(0,'config',folder.name);allseeds.append(ds)
        endpoints=fc.groupby(['seed','route']).mean(numeric_only=True).unstack('route');matches={}
        for m in ['aligned_s','target_D','other_D','error_sum','aligned_low_s']:
            q=interval(endpoints[m].acquisition-endpoints[m]['update'],rng);tol=.01 if m=='aligned_low_s' else .02;q['margin']=tol;q['pass']=bool(q['ci95'][0]>=-tol and q['ci95'][1]<=tol);matches[m]=q
        effects={m:interval(d[m],rng) for m in d.columns}
        changes={}
        for name,x,y in [('v_exchange_effect','own','v_other'),('scale_controlled_pattern_effect','v_equal_scale','v_other_equal_scale'),('norm_controlled_exchange_effect','norm_own','norm_v_other'),('encoder1_exchange_effect','own','layer_encoder1'),('encoder2_exchange_effect','own','layer_encoder2'),('heads_exchange_effect','own','layer_heads')]:
            if x in d and y in d:changes[name]=interval(d[x]-d[y],rng)
        gates={'endpoint_match_all':all(q['pass'] for q in matches.values()),'own_lower_above_005':effects['own']['ci95'][0]>.05,'v_swap_upper_below_zero':effects['v_other']['ci95'][1]<0}
        if 'reset' in effects:gates['reset_inside_002']=effects['reset']['ci95'][0]>=-.02 and effects['reset']['ci95'][1]<=.02
        if 'scale_controlled_pattern_effect' in changes:gates['pattern_effect_lower_above_zero']=changes['scale_controlled_pattern_effect']['ci95'][0]>0
        nn=nc[nc['mode'].str.startswith('norm_')];normcheck={}
        if len(nn):
            err=abs(nn.actual_norm/nn.target_norm-1);normcheck={'max_relative_error':float(err.max()),'pass':bool(err.max()<1e-4)}
            path=nn.groupby(['seed','direction','route','mode']).actual_norm.sum().unstack(['route','mode']);normcheck['max_paired_path_relative_range']=float(((path.max(axis=1)-path.min(axis=1))/path.mean(axis=1)).max())
        records[folder.name]={'n_seeds':len(d),'formation_match':matches,'effects':effects,'contrasts':changes,'gates':gates,'norm_matching':normcheck}
        path=nc.groupby(['seed','direction','route','mode']).actual_norm.sum().reset_index();path['config']=folder.name;normrecords.append(path)
    pd.concat(allseeds).to_csv(a.out/'seed_effects.csv',index=False);pd.concat(normrecords).to_csv(a.out/'path_lengths.csv',index=False)
    (a.out/'result.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n');(a.out/'input_hashes.json').write_text(json.dumps(inputs,indent=2)+'\n')
    print(json.dumps({k:{'gates':v['gates'],'effects':{m:q['mean'] for m,q in v['effects'].items()}} for k,v in records.items()},indent=2))
if __name__=='__main__':main()
