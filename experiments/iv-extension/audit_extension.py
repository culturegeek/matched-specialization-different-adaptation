"""Independent raw-CSV means, completeness, matching and frozen hash audit."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json,hashlib,math,statistics
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=ap.parse_args();R=a.root;checks=0;configs=0;maxerr=0.;allmodels=0;allrows=0
    for regname in ['registration_saved.json','registration_experiments.json']:
        for p,h in json.loads((R/regname).read_text())['sha256'].items():assert sha(R/p)==h,p;checks+=1
    for phase in ['general','beta','mechanism']:
        raw=R/f'runs_{phase}';summary=json.loads((R/f'results_{phase}/result.json').read_text())
        for p,h in json.loads((raw/'manifest.json').read_text()).items():assert sha(raw/p)==h,p;checks+=1
        for name,result in summary.items():
            configs+=1;n=40 if phase=='mechanism' else 32;expectedseeds=set(range(20000,20040)) if phase=='mechanism' else set(range(32000,32032));paths=list((raw/name).glob('seed_*'));assert {int(p.name[5:]) for p in paths}==expectedseeds
            grouped=defaultdict(list);endpoint=defaultdict(list);identities=set()
            for p in paths:
                table=rows(p/'timecourse.csv');allrows+=len(table);start={};stop={};modecount=7 if phase=='mechanism' else 6;assert len(table)==4*modecount*2
                for r in table:
                    k=(int(r['seed']),r['direction'],r['route'],r['mode']);dest=start if r['step']=='0' else stop;assert k not in dest;dest[k]=float(r['aligned_s'])
                    assert all(math.isfinite(float(r[q])) for q in ['aligned_s','target_D','other_D','error_sum','aligned_low_s'])
                assert start.keys()==stop.keys()
                for k,v in start.items():
                    seed,direction,route,mode=k;grouped[seed,mode].append((1 if route=='acquisition' else -1)*(v-stop[k])/2)
                for direction in ['A','B']:
                    for route in ['acquisition','update']:
                        vals=[v for (s,di,ro,mo),v in start.items() if di==direction and ro==route];assert max(vals)==min(vals)
                for r in rows(p/'formation_endpoints.csv'):
                    for m in result['formation_match']:endpoint[int(r['seed']),m].append((1 if r['route']=='acquisition' else -1)*float(r[m])/2)
                ns=rows(p/'update_diagnostics.csv');assert len(ns)==4*modecount*10
                controlled=defaultdict(list)
                for r in ns:
                    if r['mode'].startswith('norm_'):
                        e=abs(float(r['actual_norm'])/float(r['target_norm'])-1);assert e<1e-4;maxerr=max(maxerr,e);controlled[int(r['step']),r['direction']].append(float(r['actual_norm']))
                for vals in controlled.values():assert len(vals)==4 and (max(vals)-min(vals))/statistics.mean(vals)<2e-4
                allmodels+=4
            values={m:np.array([sum(grouped[s,m]) for s in sorted(expectedseeds)]) for m in result['effects']}
            for m,ci in result['effects'].items():assert abs(values[m].mean()-ci['mean'])<1e-12;checks+=1
            mapping={'v_exchange_effect':('own','v_other'),'scale_controlled_pattern_effect':('v_equal_scale','v_other_equal_scale'),'norm_controlled_exchange_effect':('norm_own','norm_v_other'),'encoder1_exchange_effect':('own','layer_encoder1'),'encoder2_exchange_effect':('own','layer_encoder2'),'heads_exchange_effect':('own','layer_heads')}
            for k,ci in result['contrasts'].items():x,y=mapping[k];assert abs((values[x]-values[y]).mean()-ci['mean'])<1e-12;checks+=1
            for m,ci in result['formation_match'].items():assert abs(np.mean([sum(endpoint[s,m]) for s in expectedseeds])-ci['mean'])<1e-12;checks+=1
    report={'pass':True,'checks':checks,'config_cohorts':configs,'formation_states_including_reused':allmodels,'timecourse_rows':allrows,'norm_matching_max_relative_error':maxerr,'new_unique_seeds':32,'new_setting_seed_pairs':288,'reused_checkpoint_seeds':40,'scope':'Independent CSV means, row counts, finite metrics, identical starts, norm constraints, all frozen hashes. Intervals use the separately frozen analysis scripts.'}
    (R/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
