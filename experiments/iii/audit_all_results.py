#!/usr/bin/env python3
"""実験III v0.1 の結果構造、既知の不成立、追試の対応を一括監査する。"""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).parent
EXPECTED={
 'results_a0/result.json':{},
 'results_a1/result.json':{'M_causal_model':False,'Heval_evaluation_horizon':False,'Risk_sensitivity':False},
 'results_a1_confirmatory/result.json':{},
 'results_a2/result.json':{},
 'results_a2b/result.json':{},
 'results_b1/result.json':{},
 'results_b2/result.json':{},
 'results_c/result.json':{'G3_finite_transition':False},
 'results_c_transition_confirmatory/result.json':{},
}
def main():
 for rel,failures in EXPECTED.items():
  path=ROOT/rel
  if not path.exists():raise FileNotFoundError(path)
  r=json.loads(path.read_text(encoding='utf-8'))
  if not r['completeness']['pass']:raise AssertionError(f'{rel}: completeness FAIL')
  got={k:bool(v['pass']) for k,v in r['gates'].items()}
  for key,value in failures.items():
   if got.get(key) is not value:raise AssertionError(f'{rel}: expected {key}={value}, got {got.get(key)}')
  unexpected=[k for k,v in got.items() if not v and k not in failures]
  if unexpected:raise AssertionError(f'{rel}: unexpected failures {unexpected}')
 print('AUDIT PASS: 9 result sets; known preregistered failures preserved; follow-ups present.')
if __name__=='__main__':main()
