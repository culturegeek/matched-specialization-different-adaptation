"""Independent final-estimate audit using a dense seed/direction/route tensor."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=[]
for seed in range(20000,20040):
 d=np.load(a.input/f'seed_{seed}/records.npz');v=d['values'];fields=list(d['fields']);ix={f:i for i,f in enumerate(fields)}
 assert np.all(v[:,0]==seed)
 grid=v.reshape(2,2,500,4,len(fields))
 for di in range(2):
  for ri in range(2):
   assert np.all(grid[di,ri,:,:,1]==di) and np.all(grid[di,ri,:,:,2]==ri)
   assert np.array_equal(grid[di,ri,:,:,3],np.broadcast_to(np.arange(1,501)[:,None],(500,4)))
   assert np.array_equal(grid[di,ri,:,:,4],np.broadcast_to(np.arange(4),(500,4)))
 rows.append(grid)
A=np.stack(rows)[:,:,:,499,:,:];J=json.loads(a.result.read_text());B=np.random.default_rng(2026101702).integers(0,40,(5000,40));checked=0
def check(x,target,log=False):
 global checked
 z=np.log(x) if log else x;f=np.exp if log else lambda v:v
 got=[f(z.mean()),*f(np.quantile(z[B].mean(axis=1),[.025,.975]))]
 assert np.allclose(got,[target[k] for k in ['mean','low','high']],rtol=1e-11,atol=1e-13),(got,target)
 assert target['n']==40;checked+=1
for li,layer in enumerate(['encoder1','encoder2','head_a','head_b']):
 for ri,route in enumerate(['acquisition','update']):
  x=A[:,:,ri,li,:];target=J['final'][layer][route]
  for field in ['v_after','v_initial','v_self','v_cross','v_regularization','v_rounding','inverse_scale']:check(x[:,:,ix[field]].mean(axis=1),target[field])
  check((-x[:,:,ix['v_cross']]/x[:,:,ix['v_self']]).mean(axis=1),target['cancellation'])
  for field in ['v_initial','v_self','v_cross','v_regularization','v_rounding']:check((x[:,:,ix[field]]/x[:,:,ix['v_after']]).mean(axis=1),target[field+'_fraction'])
 ratio=np.exp(np.log(A[:,:,0,li,ix['v_after']]/A[:,:,1,li,ix['v_after']]).mean(axis=1));check(ratio,J['final'][layer]['v_ratio'],True)
assert checked==108
result={'status':'PASS','final_seed_summary_intervals':checked,'raw_records_independently_reshaped':320000}
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
