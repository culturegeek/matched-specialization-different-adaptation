"""Independent raw-CSV arithmetic and intervention-integrity checks, without importing the runner."""
from pathlib import Path
import json,csv,hashlib,math,itertools
import numpy as np
HERE=Path(__file__).resolve().parent

def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 reg=json.loads((HERE/'registration.json').read_text());nchecks=0
 for p,h in reg['sha256'].items():assert sha(HERE/p)==h,p;nchecks+=1
 root=HERE/'runs_confirm';manifest=json.loads((root/'manifest.json').read_text())
 for p,h in manifest.items():assert sha(root/p)==h,p;nchecks+=1
 result=json.loads((HERE/'results_confirm/result.json').read_text());total=0
 for cap in [2,3]:
  dirs=[a+b for a in 'ABC' for b in 'ABC' if a!=b];paths=sorted((root/f'capacity_{cap}').glob('seed_*'));assert {int(p.name[5:]) for p in paths}==set(range(36000,36040));nchecks+=1;values={};forms={};originforms={};sens=[]
  for p in paths:
   seed=int(p.name[5:]);data=rows(p/'timecourse.csv');assert len(data)==432;total+=len(data);keys=set();d={}
   for r in data:
    key=(r['origin'],r['destination'],r['route'],r['mode'],int(r['step']));assert key not in keys;keys.add(key);d[key]=r
    for name in [k for k in r if k.startswith(('high_','low_'))]:assert math.isfinite(float(r[name]))
    for level in ['high','low']:
     es=[float(r[f'{level}_D_{t}']) for t in 'ABC'];assert abs(sum(es)-float(r[level+'_error_sum']))<1e-11;assert abs(sum(float(r[f'{level}_q_{t}']) for t in 'ABC'))<1e-11
     for a,b in itertools.permutations('ABC',2):assert abs((float(r[level+'_D_'+b])-float(r[level+'_D_'+a]))/(float(r[level+'_D_'+b])+float(r[level+'_D_'+a]))-float(r[level+'_s_'+a+b]))<1e-11
   modes=['own','m_other','v_other','reset','v_equal_scale','v_other_equal_scale']
   for f,j in itertools.permutations('ABC',2):
    for route in ['acquisition','update']:
     assert len({d[f,j,route,m,0]['high_s_'+f+j] for m in modes})==1
   for mode,h in itertools.product(modes,[10,50]):
    ss=[]
    for f,j in itertools.permutations('ABC',2):
     x=[]
     for route in ['acquisition','update']:x.append(float(d[f,j,route,mode,0]['high_s_'+f+j])-float(d[f,j,route,mode,h]['high_s_'+f+j]))
     ss.append(x[0]-x[1])
    values[seed,mode,h]=sum(ss)/6
   diag=rows(p/'state_diagnostics.csv');assert len(diag)==72
   for r in diag:
    if 'equal_scale' in r['mode']:assert abs(float(r['q_initial'])/float(r['q_common_target'])-1)<2e-6
   fdata=rows(p/'formation_endpoints.csv');assert len(fdata)==6;fd={(r['origin'],r['route']):r for r in fdata}
   for k in ['D0','D1','D2','error_sum','s01','s02','low_s01','low_s02']:forms[seed,k]=sum(float(fd[t,'acquisition'][k])-float(fd[t,'update'][k]) for t in 'ABC')/3
   for t in 'ABC':
    for k in ['D0','D1','D2','error_sum','s01','s02','low_s01','low_s02']:originforms[seed,t,k]=float(fd[t,'acquisition'][k])-float(fd[t,'update'][k])
   sens.append(json.loads((p/'sensor_diagnostics.json').read_text()))
   assert json.loads((p/'run.json').read_text())['data_blocks']==['train','confirm_probe','confirm_test']
   norm=rows(p/'update_diagnostics.csv');assert len(norm)==3600;assert all(float(r['update_norm'])>=0 and math.isfinite(float(r['update_norm'])) for r in norm);nchecks+=12
  seeds=range(36000,36040);ix=np.random.default_rng(2026101608).integers(0,40,size=(5000,40))
  def compare(v,ref):
   nonlocal nchecks
   a=np.array(v);z=a[ix].mean(1);truth=[a.mean(),np.quantile(z,.025),np.quantile(z,.975)]
   for q,k in zip(truth,['mean','low','high']):assert abs(q-ref[k])<1e-11,(cap,k,q,ref);nchecks+=1
  for k,v in result['by_capacity'][str(cap)]['formation_match'].items():compare([forms[s,k] for s in seeds],v)
  for origin,d in result['by_capacity'][str(cap)]['formation_by_origin'].items():
   for k,v in d['estimates'].items():compare([originforms[s,origin,k] for s in seeds],v)
   assert d['all_pass']==all(d['gates'].values())
   for k,ok in d['gates'].items():
    tol=.01 if k.startswith('low') else .02;v=d['estimates'][k];assert ok==(v['low']>-tol and v['high']<tol)
  assert result['by_capacity'][str(cap)]['all_endpoint_gates_pass']==all(v['all_pass'] for v in result['by_capacity'][str(cap)]['formation_by_origin'].values())
  for t in 'ABC':compare([sum(r[f'D_{t}_when_{j}_precise'] for j in 'ABC' if j!=t)/2-r[f'D_{t}_when_{t}_precise'] for r in sens],result['by_capacity'][str(cap)]['sensor_precision_gain'][t])
  for h in [10,50]:
   r=result['by_capacity'][str(cap)]['horizons'][str(h)]
   for m,v in r['effects'].items():compare([values[s,m,h] for s in seeds],v)
   assert r['gates']['v_exchange_effect_positive']==(r['contrasts']['v_exchange_effect']['low']>0)
   for name,a,b in [('v_exchange_effect','own','v_other'),('m_exchange_change','m_other','own'),('scale_controlled_pattern_effect','v_equal_scale','v_other_equal_scale')]:compare([values[s,a,h]-values[s,b,h] for s in seeds],r['contrasts'][name])
 spec=json.loads((HERE/'data/split_manifest.json').read_text());assert sha(HERE/'data/energydata_complete.csv')==spec['source_csv_sha256'];blocks=list(spec['splits'].values())
 for a,b in zip(blocks,blocks[1:]):assert b['start_row']-a['stop_row_exclusive']==144
 raw=rows(HERE/'data/energydata_complete.csv');assert len(raw)==19735;train=raw[:blocks[0]['stop_row_exclusive']];arr=np.array([[float(r[k]) for k in spec['columns']] for r in train]);assert np.max(abs(np.array([math.fsum(arr[:,j])/len(arr) for j in range(arr.shape[1])])-spec['normalization_mean_train_only']))<1e-12;assert np.max(abs(arr.std(0)-spec['normalization_std_train_only']))<1e-12;nchecks+=8
 report={'pass':True,'checks':nchecks,'timecourse_rows':total,'new_unique_confirm_seeds':40,'capacity_seed_pairs':80,'formation_states':480,'scope':'Raw-CSV means and bootstrap intervals, complete seed/direction/mode/time cells, identical starts, q normalization, metric identities, finite update norms, frozen hashes, room-specific endpoint gates, chronological gaps, training-only scaling, sensor gain.'};(HERE/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
