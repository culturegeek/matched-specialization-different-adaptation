"""Pre-specified V-B analysis; pilot selection uses endpoint measurements only."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from run_adaptation import TARGETS,MODES,TOLERANCES,WEIGHTS,HERE

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def read(root,name):return pd.concat([pd.read_csv(p) for p in sorted(root.glob('capacity_*/seed_*/'+name))],ignore_index=True)
def ci(x,idx):
 x=np.asarray(x,dtype=float);z=x[idx].mean(1)
 return {'mean':float(x.mean()),'low':float(np.quantile(z,.025)),'high':float(np.quantile(z,.975))}

def calibrate(root,out):
 df=read(root,'formation_endpoints.csv');result={'source_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'rule':'minimize maximum absolute mean paired endpoint difference / fixed tolerance; no adaptation data; ties choose lower weight','selected':{},'candidates':{}}
 for cap in [2,3]:
  a=df[(df.capacity==cap)&(df.route=='acquisition')].set_index(['seed','origin']);candidates=[]
  for w in WEIGHTS:
   b=df[(df.capacity==cap)&(df.route=='update')&np.isclose(df.weight,w)].set_index(['seed','origin']);x=(a[list(TOLERANCES)]-b[list(TOLERANCES)]).groupby('seed').mean();assert len(x)==8
   means=x.mean();score=max(abs(means[k])/v for k,v in TOLERANCES.items());candidates.append({'weight':w,'score':float(score),'mean_endpoint_differences':means.to_dict()})
  result['candidates'][str(cap)]=candidates;result['selected'][str(cap)]=min(candidates,key=lambda c:(c['score'],c['weight']))
 assert not out.exists();dump(out,result);print(json.dumps(result['selected'],indent=2))

def analyze(root,out):
 assert not out.exists();out.mkdir(parents=True);df=read(root,'formation_endpoints.csv');tc=read(root,'timecourse.csv');result={'statistical_unit':'seed; six ordered directions averaged within seed','bootstrap_replicates':5000,'bootstrap_seed':2026101607,'by_capacity':{}};allseed=[];allform=[];alldir=[]
 for cap in [2,3]:
  d=df[df.capacity==cap];seeds=sorted(d.seed.unique());n=len(seeds);assert n in [1,40]
  idx=np.random.default_rng(2026101607).integers(0,n,size=(5000,n));a=d[d.route=='acquisition'].set_index(['seed','origin']);b=d[d.route=='update'].set_index(['seed','origin']);fd=(a[list(TOLERANCES)]-b[list(TOLERANCES)]).groupby('seed').mean().loc[seeds]
  allform+=fd.reset_index().assign(capacity=cap).to_dict('records');fm={k:ci(fd[k],idx) for k in TOLERANCES};gates={k:fm[k]['low']>-tol and fm[k]['high']<tol for k,tol in TOLERANCES.items()}
  td=tc[tc.capacity==cap];keys=['seed','origin','destination','route','mode'];initial=td[td.step==0].set_index(keys);horizons={}
  for h in [10,50]:
   final=td[td.step==h].set_index(keys);rows=[]
   for key,r in initial.iterrows():
    seed,f,j,route,mode=key;col='high_s_'+f+j;end=final.loc[key];third=next(t for t in TARGETS if t not in [f,j]);rows.append({'seed':seed,'origin':f,'destination':j,'route':route,'mode':mode,'response':r[col]-end[col],'origin_q_change':end['high_q_'+f]-r['high_q_'+f],'destination_q_change':end['high_q_'+j]-r['high_q_'+j],'third_q_change':end['high_q_'+third]-r['high_q_'+third]})
   rd=pd.DataFrame(rows);a=rd[rd.route=='acquisition'].set_index(['seed','origin','destination','mode']);b=rd[rd.route=='update'].set_index(['seed','origin','destination','mode']);cols=['response','origin_q_change','destination_q_change','third_q_change'];diff=a[cols]-b[cols];vals=diff['response'].groupby(['seed','mode']).mean().unstack('mode').loc[seeds]
   allseed+=vals.reset_index().assign(capacity=cap,horizon=h).to_dict('records');alldir+=diff.reset_index().assign(capacity=cap,horizon=h).to_dict('records')
   effects={m:ci(vals[m],idx) for m in MODES};contrasts={k:ci(vals[a]-vals[b],idx) for k,a,b in [('v_exchange_effect','own','v_other'),('m_exchange_change','m_other','own'),('scale_controlled_pattern_effect','v_equal_scale','v_other_equal_scale')]}
   direct={f+j:{m:ci(diff.xs((f,j,m),level=['origin','destination','mode'])['response'].loc[seeds],idx) for m in MODES} for f in TARGETS for j in TARGETS if f!=j}
   q={m:{k:ci(diff.xs(m,level='mode')[k].groupby('seed').mean().loc[seeds],idx) for k in cols[1:]} for m in MODES}
   gates_response={'own_above_0.05':effects['own']['low']>.05,'v_reverses':effects['v_other']['high']<0,'reset_equivalent':effects['reset']['low']>-.02 and effects['reset']['high']<.02,'m_change_equivalent':contrasts['m_exchange_change']['low']>-.02 and contrasts['m_exchange_change']['high']<.02,'pattern_effect_positive':contrasts['scale_controlled_pattern_effect']['low']>0}
   horizons[str(h)]={'effects':effects,'contrasts':contrasts,'directions':direct,'q_vector_route_difference':q,'gates':gates_response,'decision_status':'primary pre-fixed criteria' if h==10 else 'secondary descriptive time comparison using primary reference margins'}
  result['by_capacity'][str(cap)]={'n_seeds':n,'formation_match':fm,'formation_gates':gates,'all_endpoint_gates_pass':all(gates.values()),'horizons':horizons}
 dump(out/'result.json',result);pd.DataFrame(allseed).to_csv(out/'seed_effects.csv',index=False);pd.DataFrame(allform).to_csv(out/'seed_endpoint_differences.csv',index=False);pd.DataFrame(alldir).to_csv(out/'direction_effects.csv',index=False)
 for cap,r in result['by_capacity'].items():print(json.dumps({'capacity':cap,'matched':r['all_endpoint_gates_pass'],'endpoint_failed':[k for k,v in r['formation_gates'].items() if not v],'effects10':r['horizons']['10']['effects'],'gates10':r['horizons']['10']['gates']},indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['calibrate','analyze'],required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();(calibrate if a.phase=='calibrate' else analyze)(a.input,a.out)
if __name__=='__main__':main()
