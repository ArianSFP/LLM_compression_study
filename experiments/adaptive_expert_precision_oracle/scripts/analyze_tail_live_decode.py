#!/usr/bin/env python3
"""Verify and summarize the targeted all-layer development pilot."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    identity=json.loads((a.run/'identity.json').read_text())
    completed=json.loads((a.run/'completed.json').read_text())
    assert completed['completed'] and completed['all_chosen_replays_bit_exact']
    layers=identity['layers'];steps=identity['compressed_steps'];future=identity['exact_future_steps']
    assert layers==list(range(40)) and steps>=2
    points=[];timings=[];cells=0;trajectories=0
    for request in identity['request_ids']:
        gate=json.loads((a.run/'reference_gates'/(request+'.json')).read_text())
        assert gate['logits_bit_exact'] and gate['development_capture_bit_exact'] and gate['cache_bit_identical']
        for rate in identity['rates']:
            for policy in identity['policies']:
                root=a.run/request/f'r{rate}'/policy
                trajectory=json.loads((root/'trajectory.json').read_text())
                assert trajectory['request_id']==request and trajectory['rate']==rate and trajectory['policy']==policy
                assert len(trajectory['points'])==steps+future
                assert [r['step'] for r in trajectory['points']]==list(range(steps+future))
                expected={(s,l) for s in range(steps) for l in layers}
                assert len(trajectory['choices'])==len(expected)
                assert {(r['step'],r['layer']) for r in trajectory['choices']}==expected
                for r in trajectory['points']:
                    assert all(math.isfinite(r[k]) for k in ['kl','nll','delta_nll'])
                    assert r['mode']==('all_layers_compressed' if r['step']<steps else 'exact_decode_after_compressed_history')
                    points.append(dict(request_id=request,domain=trajectory['domain'],rate=rate,policy=policy,**r))
                for selected in trajectory['choices']:
                    step,layer=selected['step'],selected['layer']
                    cell=root/f'step_{step}'/f'layer_{layer:02d}'
                    meta=json.loads((cell/'bank_meta.json').read_text())
                    choice=json.loads((cell/'choice.json').read_text())
                    digest=hashlib.sha256((cell/'bank.npz').read_bytes()).hexdigest()
                    assert digest==meta['bank_sha256']==choice['bank_sha256']
                    assert meta['terminal_labels_read'] is False
                    assert meta['factor_sha256']==identity['factor_sha256'][str(layer)]
                    assert meta['q4_routed_max_abs']<=.125
                    assert choice['request_id']==request and choice['rate']==rate and choice['policy']==policy
                    assert choice['step']==step and choice['layer']==layer and choice['chosen']==selected['kind']
                    bank=np.load(cell/'bank.npz')
                    records=json.loads(str(bank['records_json']))
                    for record in records:
                        state=bank[record['kind']+'_state']
                        assert state.shape==(8,512) and state.dtype==np.uint8 and np.max(state)<=7
                        pages=int(np.unpackbits(state).sum())
                        assert pages==record['pages'] and pages<=8*rate
                    desired={'pr13'} if policy=='pr13' else {r['kind'] for r in records}
                    assert {r['kind'] for r in choice['outcomes']}==desired
                    winner=min(choice['outcomes'],key=lambda r:(r['kl'],r['kind']!='pr13'))
                    assert winner['kind']==choice['chosen'] and winner['pages']==selected['pages']
                    if layer==39:
                        saved=json.loads((root/f'step_{step}_complete.json').read_text())
                        point=trajectory['points'][step]
                        assert all(saved[k]==point[k]==winner[k] for k in ['kl','nll','delta_nll'])
                    timings.append(meta['seconds']);cells+=1
                trajectories+=1
    assert completed['trajectories']==trajectories and completed['live_layer_cells']==cells
    frame=pd.DataFrame(points)
    frame.to_parquet(a.run/'trajectory_points.parquet',index=False)
    summaries=[]
    for (request,rate),g in frame.groupby(['request_id','rate']):
        baseline=g[g.policy=='pr13'].set_index('step')
        oracle=g[g.policy=='two_choice_oracle'].set_index('step')
        row=dict(request_id=request,domain=baseline.domain.iloc[0],rate=int(rate))
        for label,index in [('compressed',list(range(steps))),('future',list(range(steps,steps+future)))]:
            row[label+'_pr13_mean_kl']=float(baseline.loc[index].kl.mean())
            row[label+'_oracle_mean_kl']=float(oracle.loc[index].kl.mean())
            row[label+'_mean_delta_kl']=float((oracle.loc[index].kl-baseline.loc[index].kl).mean())
            row[label+'_mean_delta_nll']=float((oracle.loc[index].nll-baseline.loc[index].nll).mean())
        summaries.append(row)
    result=dict(verified=True,requests=len(identity['request_ids']),trajectories=trajectories,live_layer_cells=cells,
        median_cpu_bank_seconds=float(np.median(timings)),p95_cpu_bank_seconds=float(np.quantile(timings,.95)),
        summaries=summaries,interpretation='Two selected development cases; fresh routed-expert allocations at all 40 layers; two-choice myopic oracle, not global optimization or a runtime controller. Future steps are exact after compressed history.')
    (a.run/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
