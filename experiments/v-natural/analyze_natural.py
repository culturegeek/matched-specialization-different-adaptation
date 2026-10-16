"""Pre-specified V-B analysis; pilot selection uses endpoint measurements only."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from run_natural import B,WEIGHTS,HERE
TARGETS,MODES,TOLERANCES=B.TARGETS,B.MODES,B.TOLERANCES

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def read(root,name):return pd.concat([pd.read_csv(p) for p in sorted(root.glob('capacity_*/seed_*/'+name))],ignore_index=True)
def ci(x,idx):
 x=np.asarray(x,dtype=float);z=x[idx].mean(1)
 return {'mean':float(x.mean()),'low':float(np.quantile(z,.025)),'high':float(np.quantile(z,.975))}

def calibrate(root,out):
 df=read(root,'formation_endpoints.csv');result={'source_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'rule':'per capacity and origin, minimize maximum absolute mean paired endpoint difference / tolerance; no adaptation or confirm evaluation used','selected':{},'candidates':{}}
 for cap in [2,3]:
  result['selected'][str(cap)]={};result['candidates'][str(cap)]={}
  for f in TARGETS:
   a=df[(df.capacity==cap)&(df.route=='acquisition')&(df.origin==f)].set_index('seed');candidates=[]
   for w in WEIGHTS:
    b=df[(df.capacity==cap)&(df.route=='update')&(df.origin==f)&np.isclose(df.weight,w)].set_index('seed');x=a[list(TOLERANCES)]-b[list(TOLERANCES)];assert len(x)==8
    means=x.mean();score=max(abs(means[k])/v for k,v in TOLERANCES.items());candidates.append({'weight':w,'score':float(score),'mean_endpoint_differences':means.to_dict()})
   result['candidates'][str(cap)][f]=candidates;result['selected'][str(cap)][f]=min(candidates,key=lambda x:(x['score'],x['weight']))
 assert not out.exists();dump(out,result)
 print(json.dumps({c:{f:{k:r[k] for k in ['weight','score']} for f,r in d.items()} for c,d in result['selected'].items()},indent=2))

def analyze(root,out):
 assert not out.exists();out.mkdir(parents=True);df=read(root,'formation_endpoints.csv');tc=read(root,'timecourse.csv');result={'statistical_unit':'seed; six ordered directions averaged within seed','bootstrap_replicates':5000,'bootstrap_seed':2026101608,'by_capacity':{}};allseed=[];allform=[];alldir=[]
 for cap in [2,3]:
  d=df[df.capacity==cap];seeds=sorted(d.seed.unique());n=len(seeds);assert n in [1,40]
  idx=np.random.default_rng(2026101608).integers(0,n,size=(5000,n));a=d[d.route=='acquisition'].set_index(['seed','origin']);b=d[d.route=='update'].set_index(['seed','origin']);fd=(a[list(TOLERANCES)]-b[list(TOLERANCES)]).groupby('seed').mean().loc[seeds]
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
 # Natural rooms are not exchangeable: endpoint gates must also pass separately by origin.
 all_origin=[]
 for cap in [2,3]:
  r=result['by_capacity'][str(cap)];r['pooled_endpoint_gates_pass']=r['all_endpoint_gates_pass'];r['formation_by_origin']={};idx=np.random.default_rng(2026101608).integers(0,r['n_seeds'],size=(5000,r['n_seeds']))
  for f in TARGETS:
   d=df[(df.capacity==cap)&(df.origin==f)];a=d[d.route=='acquisition'].set_index('seed');b=d[d.route=='update'].set_index('seed');x=(a[list(TOLERANCES)]-b[list(TOLERANCES)]).sort_index();all_origin+=x.reset_index().assign(capacity=cap,origin=f).to_dict('records');est={k:ci(x[k],idx) for k in TOLERANCES};gates={k:est[k]['low']>-tol and est[k]['high']<tol for k,tol in TOLERANCES.items()};r['formation_by_origin'][f]={'estimates':est,'gates':gates,'all_pass':all(gates.values())}
  r['all_endpoint_gates_pass']=all(v['all_pass'] for v in r['formation_by_origin'].values())
  for h in ['10','50']:r['horizons'][h]['gates']['v_exchange_effect_positive']=r['horizons'][h]['contrasts']['v_exchange_effect']['low']>0
  sensor=[]
  for p in sorted((root/f'capacity_{cap}').glob('seed_*/sensor_diagnostics.json')):
   d=json.loads(p.read_text());sensor.append({f:np.mean([d[f'D_{f}_when_{j}_precise'] for j in TARGETS if j!=f])-d[f'D_{f}_when_{f}_precise'] for f in TARGETS})
  r['sensor_precision_gain']={f:ci([d[f] for d in sensor],idx) for f in TARGETS}
 result['dataset']='UCI Appliances Energy Prediction, DOI 10.24432/C5VC8G; three room temperature/humidity pairs';result['uncertainty_scope']='Seed variability conditional on one building and fixed chronological blocks; not independent building/time draws.'
 dump(out/'result.json',result);pd.DataFrame(allseed).to_csv(out/'seed_effects.csv',index=False);pd.DataFrame(allform).to_csv(out/'seed_endpoint_differences.csv',index=False);pd.DataFrame(all_origin).to_csv(out/'origin_endpoint_differences.csv',index=False);pd.DataFrame(alldir).to_csv(out/'direction_effects.csv',index=False)
 for cap,r in result['by_capacity'].items():print(json.dumps({'capacity':cap,'all_24_endpoint_gates_pass':r['all_endpoint_gates_pass'],'endpoint_failed':{f:[k for k,v in d['gates'].items() if not v] for f,d in r['formation_by_origin'].items()},'effects10':r['horizons']['10']['effects'],'contrasts10':r['horizons']['10']['contrasts'],'gates10':r['horizons']['10']['gates']},indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['calibrate','analyze'],required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();(calibrate if a.phase=='calibrate' else analyze)(a.input,a.out)
if __name__=='__main__':main()
