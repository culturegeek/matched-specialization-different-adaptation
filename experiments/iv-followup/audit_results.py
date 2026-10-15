"""Independent structural/numerical audit: no experiment or analysis module imports."""
import csv
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',default='results_confirm')
    out=ROOT/parser.parse_args().folder
    reg=json.loads((ROOT/'registration.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in reg['file_hashes'].items())
    modes=reg['modes']; long={'own','m_other','v_other','reset','sgd_calibrated'}
    expected_steps=set(range(11))|{20,50,100,200,500}
    result=json.loads((out/'result.json').read_text())
    maximum_clone_diff=0.;prefix_diff=0.;n_rows=0;n_updates=0
    diffs=defaultdict(list);sgd_checks=[]
    for seed in reg['seeds']:
        folder=out/f'seed_{seed}'
        rows=read(folder/'timecourse.csv');n_rows+=len(rows)
        cells=defaultdict(dict)
        for r in rows:
            for name in ['aligned_s','D_A','D_B','aligned_low_s']:
                assert np.isfinite(float(r[name]))
            t=int(r['step']);key=(r['route'],r['direction'],r['mode'],r['trajectory'])
            assert t not in cells[key]
            cells[key][t]=r
        assert len(cells)==4*(len(modes)+2)
        for route in ['acquisition','update']:
            for d in ['A','B']:
                starts=[]
                for mode in modes:
                    c=cells[(route,d,mode,'continued_opposition')]
                    assert set(c)==(expected_steps if mode in long else set(range(11)))
                    starts.append(float(c[0]['aligned_s']))
                maximum_clone_diff=max(maximum_clone_diff,max(starts)-min(starts))
                for mode in ['own','reset']:
                    neutral=cells[(route,d,mode,'pulse_then_neutral')]
                    assert set(neutral)==expected_steps|{510}
                    cont=cells[(route,d,mode,'continued_opposition')]
                    prefix_diff=max(prefix_diff,max(abs(float(neutral[t]['aligned_s'])-float(cont[t]['aligned_s'])) for t in range(11)))
        for mode in modes:
            mirror=[]
            for d in ['A','B']:
                ca=cells[('acquisition',d,mode,'continued_opposition')]
                cu=cells[('update',d,mode,'continued_opposition')]
                mirror.append((float(ca[0]['aligned_s'])-float(ca[10]['aligned_s']))-(float(cu[0]['aligned_s'])-float(cu[10]['aligned_s'])))
            diffs[mode].append(sum(mirror)/2)
        updates=read(folder/'update_diagnostics.csv');n_updates+=len(updates)
        assert len(updates)==4*(5*500+5*10+2*510)
        first={}
        for r in updates:
            for name in ['gradient_norm','update_norm','m_norm','v_sum']:
                assert np.isfinite(float(r[name])) and float(r[name])>=0
            if r['mode'].startswith('sgd'):
                assert float(r['m_norm'])==float(r['v_sum'])==0
            if r['trajectory']=='continued_opposition' and r['step']=='1':
                first[(r['route'],r['direction'],r['mode'])]=float(r['update_norm'])
        for d in ['A','B']:
            ratios=[first[(route,d,'sgd_calibrated')]/first[(route,d,'own')] for route in ['acquisition','update']]
            sgd_checks.append(abs(np.sqrt(ratios[0]*ratios[1])-1))
    assert maximum_clone_diff==0 and prefix_diff==0
    assert max(sgd_checks)<1e-4
    verified={}
    rng=np.random.default_rng(1032026)
    for mode,v in diffs.items():
        x=np.array(v);boot=x[rng.integers(0,len(x),(5000,len(x)))].mean(axis=1)
        lo,hi=np.quantile(boot,[.025,.975]);reported=result['short_route_effects'][mode]
        assert abs(x.mean()-reported['mean'])<1e-12
        # Separate bootstrap stream need not give identical interval endpoints.
        assert max(abs(lo-reported['low']),abs(hi-reported['high']))<.005
        verified[mode]={'mean':float(x.mean()),'low':float(lo),'high':float(hi)}
    old=read(ROOT/'vendor/experiment_iv_complete_v05/IV-E/results_confirm/timecourse.csv')
    old={(r['route'],r['direction'],r['optimizer_component'],r['step']):float(r['aligned_s']) for r in old if r['seed']=='9100' and r['protocol']=='combined_switch'}
    rep=read(ROOT/'results_reproduce/seed_9100/timecourse.csv')
    repro=max(abs(float(r['aligned_s'])-old[(r['route'],r['direction'],r['mode'],r['step'])]) for r in rep)
    assert repro==0
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in {'audit.json','manifest.json'}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report={'pass':True,'seeds':len(reg['seeds']),'timecourse_rows':n_rows,'update_rows':n_updates,
            'maximum_clone_start_difference':maximum_clone_diff,'maximum_shared_prefix_difference':prefix_diff,
            'legacy_9100_reproduction_max_difference':repro,'sgd_first_step_pair_geomean_relative_error_max':max(sgd_checks),
            'independent_route_intervals':verified,'registration_sha256':sha(ROOT/'registration.json'),
            'manifest_sha256':sha(out/'manifest.json')}
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
