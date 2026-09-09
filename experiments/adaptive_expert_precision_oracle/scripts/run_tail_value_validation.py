#!/usr/bin/env python3
"""Frozen-metric validation, adding metric choices to the private-cache probe."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import run_tail_value_3090 as base

POLICIES=['pr13','current_token_oracle','metric','metric_d1_nonworsening']


def add_metric_probe(model,row,prefix,ref,baseline,bank,outcomes,output,metric,fit_dir):
    tokens,pos=row['prompt_token_ids'],row['decode_position']
    horizon=min(4,len(tokens)-pos-2)
    if horizon<1:
        return
    factor=fit_dir/'average_rate_factor_layer_6.npz'
    fitted=metric['layers']['6']
    if hashlib.sha256(factor.read_bytes()).hexdigest()!=fitted['factor_sha256']:
        raise ValueError('metric continuation factor changed')
    proxy=np.load(factor)['proxy']
    def score(record):
        delta=np.asarray(bank[record['key']+'_delta'])
        x=np.asarray([record['local_damage'],*np.square(delta@proxy)])
        return float(x@fitted['coefficients']),record['key']
    refs=[ref.logits.detach().clone()]
    cache=copy.deepcopy(ref.past_key_values)
    for step in range(1,horizon+1):
        value=base.forward(model,[tokens[pos+step]],cache,pos+step)
        cache=value.past_key_values
        refs.append(value.logits.detach().clone())
    del value,cache
    probe=json.loads(output.read_text())
    for rate in [360,725]:
        feasible=[r for r in outcomes if r['rate']==rate and r['kind']!='external_pr13_high']
        pr13=next(r for r in feasible if r['kind']=='pr13')
        choices={'metric':min(feasible,key=score),
                 'metric_d1_nonworsening':min([r for r in feasible if r['d1_missing']<=pr13['d1_missing']],key=score)}
        for policy,record in choices.items():
            delta=torch.tensor(bank[record['key']+'_delta'],device='cuda',dtype=torch.float32)
            def inject(m,args,out):
                if not torch.equal(out,baseline):
                    raise RuntimeError('metric cache probe baseline differs')
                return (out.float()+delta).to(out.dtype)
            handle=model.model.language_model.layers[6].mlp.experts.register_forward_hook(inject)
            try:
                candidate=base.forward(model,[tokens[pos]],copy.deepcopy(prefix),pos)
            finally:
                handle.remove()
            for step in range(horizon+1):
                if step:
                    candidate=base.forward(model,[tokens[pos+step]],candidate.past_key_values,pos+step)
                probe['outcomes'].append(dict(rate=rate,policy=policy,step=step,selected_key=record['key'],
                    **base.quality(refs[step],candidate.logits,tokens[pos+step+1])))
            del candidate
    probe['policy_names']=POLICIES
    probe['metric_selection_reads_terminal_kl']=False
    base.atomic_json(output,probe)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--metric',type=Path,required=True)
    parser.add_argument('--fit-dir',type=Path,default=Path('/workspace/pr13_average_rate_all_layers_20260822_v1/results/fit'))
    a,remaining=parser.parse_known_args()
    base_args=argparse.ArgumentParser(add_help=False)
    base_args.add_argument('--output',type=Path,required=True)
    base_args.add_argument('--manifest',type=Path,required=True)
    b,_=base_args.parse_known_args(remaining)
    requests=[json.loads(l) for l in b.manifest.read_text().splitlines()]
    if any(r['split']!='validation' for r in requests):
        raise ValueError('validation wrapper accepts validation only')
    metric=json.loads(a.metric.read_text())
    protocol=dict(wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  metric_sha256=hashlib.sha256(a.metric.read_bytes()).hexdigest(),cache_probe_policies=POLICIES,
                  model_frozen_before_validation=True,metric_predictions_read_terminal_kl=False)
    target=b.output/'validation_protocol.json'
    if not target.exists() and list((b.output/'labels').glob('*.json')):
        raise ValueError('cannot add frozen-metric protocol after validation labels')
    if target.exists() and json.loads(target.read_text())!=protocol:
        raise ValueError('validation protocol changed')
    base.atomic_json(target,protocol)
    original=base.cache_probe
    def probe(*args):
        original(*args)
        add_metric_probe(*args,metric=metric,fit_dir=a.fit_dir)
    base.cache_probe=probe
    sys.argv=[sys.argv[0],*remaining]
    base.main()


if __name__=='__main__':
    main()
