from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import csv,io,json,hashlib
import numpy as np
r=Path(__file__).resolve().parents[1]
f=r/'experiments/iv-followup'
iv=r/'experiments/iv'
checks=[]
def check(name,values,reported,digits=4):
 value=float(np.mean(values));passed=abs(value-reported)<=0.50001*10**(-digits)
 checks.append(dict(quantity=name,raw_mean=value,manuscript_value=reported,rounding_digits=digits,pass_check=passed))
 assert passed,(name,value,reported)
def route(rows,pred,field,first='acquisition',second='update'):
 d=defaultdict(list)
 for x in rows:
  if pred(x):d[(int(x['seed']),x['route'])].append(float(x[field]))
 seeds=sorted({k[0] for k in d});assert len(seeds)==40
 return [np.mean(d[(s,first)])-np.mean(d[(s,second)]) for s in seeds]
if True:
 def read(study,name):
  with (iv/study/'results_confirm'/f'{name}.csv').open() as h:return list(csv.DictReader(h))
 d=read('IV-D','protocol_summary'); forms=read('IV-D','formation_endpoints')
 check('IV-D matched aligned allocation',route(forms,lambda x:True,'aligned_s'),-.0064)
 for mode,vals in [('carried',[.1668,.1510,.1951]),('reset',[.0011,.0120,.0103])]:
  for protocol,v in zip(['acquisition_pulse','update_pulse','combined_switch'],vals):
   check('IV-D '+mode+' '+protocol,route(d,lambda x:x['optimizer_mode']==mode and x['protocol']==protocol,'pulse_drop'),v)
 for study,rs,vs in [('IV-E',('acquisition','update'),[.1968,.1982,-.1265,.0097]),('IV-E-update-bias-control',('constant','variable'),[.0324,.0321,-.0141,.0046])]:
  data=read(study,'protocol_summary')
  for mode,v in zip(['own','m_other','v_other','reset'],vs):
   check(study+' '+mode,route(data,lambda x:x['optimizer_component']==mode and x['protocol']=='combined_switch','pulse_drop',*rs),v)
 rows=[x for x in d if x['optimizer_mode']=='carried' and x['protocol']=='combined_switch' and x['route']=='acquisition']
 check('IV-D acquisition zero crossing percent',[100*float(x['switched_during_pulse']) for x in rows],67.5,1)
follow=[]
for path in sorted((f/'results_confirm').glob('seed_*/timecourse.csv')):
 with path.open() as h:follow.extend(csv.DictReader(h))
assert len(follow)==27040
start={(x['seed'],x['route'],x['direction'],x['mode'],x['trajectory']):float(x['aligned_s']) for x in follow if int(x['step'])==0}
for x in follow:x['drop']=start[(x['seed'],x['route'],x['direction'],x['mode'],x['trajectory'])]-float(x['aligned_s'])
for mode,step,value in [('own',10,.1907),('v_equal_scale',10,.0875),('scalar_v_own',10,.1251),('scalar_v_equal',10,.0395),('v_other_equal_scale',10,-.0110),('reset',10,.0096),('reset',50,.0274),('sgd_calibrated',10,.0123),('sgd_calibrated',50,.0433),('sgd_calibrated',100,.0309)]:
 check(f'Follow-up {mode} step {step}',route(follow,lambda x:x['mode']==mode and int(x['step'])==step and x['trajectory']=='continued_opposition','drop'),value)
check('Follow-up own endpoint difference at 500',route(follow,lambda x:x['mode']=='own' and int(x['step'])==500 and x['trajectory']=='continued_opposition','aligned_s'),-.00045,5)
post=json.loads((f/'posthoc_review_results.json').read_text())['endpoint_vs_drop_difference']['continued_opposition']['500']['endpoint_allocation_difference']
assert [round(v,5) for v in post['ci95']]==[-.00146,.00057]
manifest=json.loads((f/'results_confirm/manifest.json').read_text())
missing=[];bad=[]
for name,expected in manifest.items():
 path=f/name
 if not path.exists():missing.append(name);continue
 if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:bad.append(name)
assert not missing and not bad,(missing,bad)
record={'date':'2026-10-15','scope':'Independent arithmetic reaggregation of selected central means from archived CSVs; no training and no new inferential tests.', 'checks':checks,'all_pass':True,'frozen_manifest_files_checked':len(manifest),'endpoint_interval_source':str((f/'posthoc_review_results.json').relative_to(r)),'interval_note':'The canonical confirmatory bootstrap uses a separate random stream and gives [-0.001419,0.000582]; the manuscript retains the existing separately labeled post-hoc interval [-0.001462,0.000568]. Both use the same endpoint mean. Figure intervals likewise use their documented plotting bootstrap, not a replacement confirmatory test.','input_sha256':{str(x.relative_to(r)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [f/'results_confirm/manifest.json',f/'posthoc_review_results.json',f/'posthoc_update_norm_results.json']}}
out=r/'generated/core_numeric_audit.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'all_pass':True,'frozen_files_checked':len(manifest)},ensure_ascii=False))
