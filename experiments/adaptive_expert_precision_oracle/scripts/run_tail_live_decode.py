#!/usr/bin/env python3
"""Small live all-layer oracle: fresh allocations and private evolving caches."""
import argparse
import copy
from collections import deque
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import run_tail_value_3090 as base
import tail_live_options as live


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npz(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f,**values)
    temporary.replace(path)


def execute(model,token,position,inherited,mapping):
    handles=[]
    for layer,(baseline,delta) in mapping.items():
        expected=torch.tensor(baseline,device='cuda',dtype=torch.bfloat16)
        change=torch.tensor(delta,device='cuda',dtype=torch.float32)
        def inject(module,args,out,expected=expected,change=change):
            if not torch.equal(out,expected):
                raise RuntimeError('live prefix allocation baseline changed')
            return (out.float()+change).to(out.dtype)
        handles.append(model.model.language_model.layers[layer].mlp.experts.register_forward_hook(inject))
    try:
        return base.forward(model,[token],copy.deepcopy(inherited),position)
    finally:
        for handle in handles:
            handle.remove()


def initialize_worker():
    torch.set_num_threads(1)
    live._CACHE_LIMIT=1*2**30


def prepare_worker(cap,layer,rate,checkpoint,trees,fit_dir,factor):
    return live.prepare(cap,layer,rate,checkpoint,trees,fit_dir,factor)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--development-run',type=Path,required=True)
    p.add_argument('--validation-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('/workspace/qwen36_mxfp4_candidate'))
    p.add_argument('--trees',type=Path,default=Path('/workspace/codebook_granularity_study/locked/selected_trees.json'))
    p.add_argument('--fit-dir',type=Path,default=Path('/workspace/pr13_average_rate_all_layers_20260822_v1/results/fit'))
    p.add_argument('--rates',default='360,725')
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--future',type=int,default=4)
    a=p.parse_args()
    if not json.loads((a.validation_run/'verification.json').read_text())['verified']:
        raise ValueError('finish validation verification before starting this memory-intensive pilot')
    rows=[json.loads(line) for line in a.manifest.read_text().splitlines()]
    if any(r['split']!='development' for r in rows):
        raise ValueError('live pilot uses development only')
    selected_rows=[next(r for r in rows if r['domain']==domain) for domain in ['code','dialogue_instruction']]
    rates=list(map(int,a.rates.split(',')))
    if not rates or len(set(rates))!=len(rates) or not set(rates)<={360,725}:
        raise ValueError('invalid pilot rates')
    if a.steps<2 or a.future<1 or any(r['decode_position']+a.steps+a.future>=len(r['prompt_token_ids']) for r in selected_rows):
        raise ValueError('insufficient continuation or invalid horizon')
    a.output.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).resolve().parent
    factors={str(l):sha(a.fit_dir/f'average_rate_factor_layer_{l}.npz') for l in range(40)}
    identity=dict(schema='tail_live_decode_v1',manifest_sha256=sha(a.manifest),request_ids=[r['request_id'] for r in selected_rows],
        rates=rates,compressed_steps=a.steps,exact_future_steps=a.future,layers=list(range(40)),
        policies=['pr13','two_choice_oracle'],candidate_family=['pr13','exact_local'],
        cpu_workers=8,per_worker_expert_cache_limit_bytes=1*2**30,
        factor_sha256=factors,trees_sha256=sha(a.trees),
        checkpoint_config_sha256=sha(a.checkpoint/'config.json'),
        checkpoint_index_sha256=sha(a.checkpoint/'model.safetensors.index.json'),
        pr13_config_sha256=sha(scripts.parent/'configs/qwen36_mxfp4_average_rate_all_layers.json'),
        parent_commit='8cd2230a21bbe4aca2910ec7fb6b5426bd43eda7',
        code_sha256={name:sha(scripts/name) for name in ['run_tail_live_decode.py','tail_live_options.py',
            'prepare_tail_value_bank.py','tail_value_streaming.py','run_tail_value_3090.py']},
        scope='post-development exploratory; no validation outcomes used for selection; no global oracle or runtime claim')
    ip=a.output/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:
        raise ValueError('live pilot resume identity changed')
    base.atomic_json(ip,identity)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter()
    model=base.load_streamed(a.checkpoint)
    if len(model.model.language_model.layers)!=40:
        raise ValueError('layer count changed')
    capture,handles=base.record_observers(model,list(range(40)))
    cases=[]
    for row in selected_rows:
        tokens,pos=row['prompt_token_ids'],row['decode_position']
        pref=base.forward(model,tokens[:pos])
        prefix=pref.past_key_values
        del pref
        first=base.forward(model,[tokens[pos]],copy.deepcopy(prefix),pos)
        second=base.forward(model,[tokens[pos]],copy.deepcopy(prefix),pos)
        gate=base.cache_state_metrics(first.past_key_values,second.past_key_values).to_dict()
        if not torch.equal(first.logits,second.logits) or not gate['cache_bit_identical']:
            raise RuntimeError('live reference zero gate failed')
        stored=np.load(a.development_run/'captures'/(row['request_id']+'.npz'))
        if any(not np.array_equal(stored[k],capture[k].numpy()) for k in stored.files):
            raise RuntimeError('live reference differs from saved development capture')
        base.atomic_json(a.output/'reference_gates'/(row['request_id']+'.json'),dict(logits_bit_exact=True,development_capture_bit_exact=True,**gate))
        del second
        refs=[first.logits.detach().clone()]
        refcache=first.past_key_values
        for step in range(1,a.steps+a.future):
            ref=base.forward(model,[tokens[pos+step]],refcache,pos+step)
            refcache=ref.past_key_values
            refs.append(ref.logits.detach().clone())
        del first,ref,refcache
        cases.append(dict(row=row,tokens=tokens,pos=pos,prefix=prefix,refs=refs))
    trajectories=[]
    ready=deque()
    pending={}
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker) as pool:
        def schedule(state):
            rate,policy,step,layer=(state[k] for k in ['rate','policy','step','layer'])
            case=state['case'];tokens,pos=case['tokens'],case['pos']
            current=execute(model,tokens[pos+step],pos+step,state['inherited'],state['mapping'])
            cap={f'{name}_{layer}':capture[f'{name}_{layer}'].numpy().copy() for name in ['x','routed','ids','scores']}
            del current
            cell=a.output/case['row']['request_id']/f'r{rate}'/policy/f'step_{step}'/f'layer_{layer:02d}'
            item=(state,cap,cell)
            if (cell/'bank.npz').exists() and (cell/'bank_meta.json').exists():
                ready.append(item)
            else:
                future=pool.submit(prepare_worker,cap,layer,rate,a.checkpoint,a.trees,a.fit_dir,factors[str(layer)])
                pending[future]=item

        for case in cases:
            for rate in rates:
                for policy in ['pr13','two_choice_oracle']:
                    schedule(dict(case=case,rate=rate,policy=policy,step=0,layer=0,mapping={},
                                  inherited=copy.deepcopy(case['prefix']),points=[],choices=[]))
        while pending or ready:
            if not ready:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    state,cap,cell=pending.pop(future)
                    options,facts=future.result()
                    values=dict(cap)
                    records=[]
                    for option in options:
                        kind=option['kind']
                        values[kind+'_state']=option['state']
                        values[kind+'_delta']=option['delta']
                        records.append({k:v for k,v in option.items() if k not in ['state','delta']})
                    values['records_json']=np.asarray(json.dumps(records))
                    save_npz(cell/'bank.npz',values)
                    base.atomic_json(cell/'bank_meta.json',dict(bank_sha256=sha(cell/'bank.npz'),
                                                               terminal_labels_read=False,**facts))
                    ready.append((state,cap,cell))
            state,cap,cell=ready.popleft()
            rate,policy,step,layer=(state[k] for k in ['rate','policy','step','layer'])
            case=state['case'];row=case['row'];tokens,pos,refs=case['tokens'],case['pos'],case['refs']
            position=pos+step
            bp,mp,cp=cell/'bank.npz',cell/'bank_meta.json',cell/'choice.json'
            meta=json.loads(mp.read_text())
            if sha(bp)!=meta['bank_sha256'] or meta['factor_sha256']!=factors[str(layer)]:
                raise ValueError('live bank seal changed')
            bank=np.load(bp)
            if any(not np.array_equal(cap[k],bank[k]) for k in cap):
                raise RuntimeError('saved live bank input differs from current trajectory')
            records=json.loads(str(bank['records_json']))
            if cp.exists():
                choice=json.loads(cp.read_text())
                if choice['bank_sha256']!=meta['bank_sha256'] or choice['request_id']!=row['request_id'] or any(choice[k]!=state[k] for k in ['rate','policy','step','layer']):
                    raise ValueError('live choice identity changed')
            else:
                evaluated=[]
                for record in records:
                    if policy=='pr13' and record['kind']!='pr13':
                        continue
                    kind=record['kind']
                    trial=dict(state['mapping'])
                    trial[layer]=(cap[f'routed_{layer}'],bank[kind+'_delta'])
                    candidate=execute(model,tokens[position],position,state['inherited'],trial)
                    quality=base.quality(refs[step],candidate.logits,tokens[position+1])
                    if not all(math.isfinite(v) for v in quality.values()):
                        raise RuntimeError('nonfinite live quality')
                    evaluated.append(dict(record,**quality))
                    del candidate
                winner=min(evaluated,key=lambda r:(r['kl'],r['kind']!='pr13'))
                choice=dict(bank_sha256=meta['bank_sha256'],request_id=row['request_id'],chosen=winner['kind'],outcomes=evaluated,
                            rate=rate,policy=policy,step=step,layer=layer)
                base.atomic_json(cp,choice)
            kind=choice['chosen']
            selected=bank[kind+'_state']
            pages=live.frozen.state_page_count(selected)
            if pages>8*rate:
                raise RuntimeError('live selected state exceeds payload')
            state['mapping'][layer]=(cap[f'routed_{layer}'],bank[kind+'_delta'].copy())
            current=execute(model,tokens[position],position,state['inherited'],state['mapping'])
            quality=base.quality(refs[step],current.logits,tokens[position+1])
            expected=next(r for r in choice['outcomes'] if r['kind']==kind)
            if quality['kl']!=expected['kl'] or quality['nll']!=expected['nll']:
                raise RuntimeError('live chosen candidate replay differs')
            state['choices'].append(dict(step=step,layer=layer,kind=kind,pages=pages))
            print('live',row['request_id'],rate,policy,step,layer,kind,quality['kl'],flush=True)
            if layer<39:
                state['layer']+=1
                del current
                schedule(state)
                continue
            state['inherited']=current.past_key_values
            state['points'].append(dict(step=step,mode='all_layers_compressed',**quality))
            del current
            base.atomic_json(a.output/row['request_id']/f'r{rate}'/policy/f'step_{step}_complete.json',state['points'][-1])
            if step+1<a.steps:
                state.update(step=step+1,layer=0,mapping={})
                schedule(state)
                continue
            for future_step in range(a.steps,a.steps+a.future):
                current=base.forward(model,[tokens[pos+future_step]],state['inherited'],pos+future_step)
                state['inherited']=current.past_key_values
                state['points'].append(dict(step=future_step,mode='exact_decode_after_compressed_history',
                    **base.quality(refs[future_step],current.logits,tokens[pos+future_step+1])))
                del current
            trajectory=dict(rate=rate,policy=policy,request_id=row['request_id'],domain=row['domain'],
                            points=state['points'],choices=state['choices'])
            base.atomic_json(a.output/row['request_id']/f'r{rate}'/policy/'trajectory.json',trajectory)
            trajectories.append(trajectory)
            del state['inherited']
    for h in handles:
        h.remove()
    base.atomic_json(a.output/'completed.json',dict(completed=True,trajectories=len(trajectories),
        all_chosen_replays_bit_exact=True,
        live_layer_cells=sum(len(t['choices']) for t in trajectories),elapsed_seconds=time.perf_counter()-started,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,cpu_workers=8,
        per_worker_expert_cache_limit_bytes=1*2**30))


if __name__=='__main__':
    main()
