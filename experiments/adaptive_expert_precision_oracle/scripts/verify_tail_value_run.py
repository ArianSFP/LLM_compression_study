#!/usr/bin/env python3
"""Fail closed on incomplete cells, changed banks, zero gates, or cache probes."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def verify(run, manifest):
    identity=json.loads((run/'identity.json').read_text())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==identity['manifest_sha256']
    requests={r['request_id']:r for r in map(json.loads,manifest.read_text().splitlines())}
    completion=json.loads((run/'completed.json').read_text())
    assert completion['completed'] is True
    seal=json.loads((run/'bank_seal.json').read_text())
    expected={f"{r}_l{l}" for r in identity['request_ids'] for l in identity['layers']}
    assert set(seal)=={s+'.npz' for s in expected}
    assert {p.stem for p in (run/'labels').glob('*.json')}==expected
    domain_first={}
    candidate_rows=0
    for request in identity['request_ids']:
        zero=json.loads((run/'zero_gates'/(request+'.json')).read_text())
        assert zero['logits_bit_exact'] and zero['cache_bit_identical']
        for layer in identity['layers']:
            name=f'{request}_l{layer}'
            cell=json.loads((run/'labels'/(name+'.json')).read_text())
            assert cell['request_id']==request and cell['layer']==layer
            assert cell['split']!='reserve'
            domain_first.setdefault(cell['domain'],request)
            digest=hashlib.sha256((run/'banks'/(name+'.npz')).read_bytes()).hexdigest()
            assert digest==seal[name+'.npz']==cell['bank_sha256']
            for rate in [360,725]:
                records=[r for r in cell['outcomes'] if r['rate']==rate]
                assert sum(r['kind']=='pr13' for r in records)==1
                assert sum(r['kind']=='external_pr13_high' for r in records)==1
                assert all(r['pages']<=8*rate for r in records if r['kind']!='external_pr13_high')
                assert all(math.isfinite(r['kl']) and math.isfinite(r['nll']) for r in records)
            candidate_rows+=len(cell['outcomes'])
    probes=0
    omitted=[]
    policies=['pr13','current_token_oracle']
    if (run/'validation_protocol.json').exists():
        policies=json.loads((run/'validation_protocol.json').read_text())['cache_probe_policies']
    if 6 in identity['layers']:
        for request in domain_first.values():
            row=requests[request]
            horizon=min(4,len(row['prompt_token_ids'])-row['decode_position']-2)
            if horizon<1:
                omitted.append(dict(request_id=request,domain=row['domain'],reason='no continuation tokens after the fixed decode position'))
                continue
            probe=json.loads((run/'cache_probe'/(request+'.json')).read_text())
            assert probe['horizon']==horizon
            expected_probe={(rate,policy,step) for rate in [360,725] for policy in policies for step in range(horizon+1)}
            assert len(probe['outcomes'])==len(expected_probe)
            assert {(r['rate'],r['policy'],r['step']) for r in probe['outcomes']}==expected_probe
            cell=json.loads((run/'labels'/f'{request}_l6.json').read_text())
            labels={r['key']:r for r in cell['outcomes']}
            for outcome in probe['outcomes']:
                if outcome['step']==0:
                    reference=labels[outcome['selected_key']]
                    assert outcome['kl']==reference['kl'] and outcome['nll']==reference['nll']
            probes+=1
    result=dict(verified=True,requests=len(identity['request_ids']),cells=len(expected),
                candidate_rows=candidate_rows,cache_probes=probes,omitted_cache_probes=omitted)
    (run/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(verify(a.run,a.manifest),indent=2))
