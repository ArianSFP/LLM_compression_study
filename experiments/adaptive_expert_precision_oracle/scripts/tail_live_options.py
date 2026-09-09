"""Fresh PR13/exact-local options for a live current-layer activation."""
import argparse
import copy
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits
import prepare_tail_value_bank as frozen

TREE_SHA='da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8'
_CACHE=OrderedDict()
_CACHE_BYTES=0
_CACHE_LIMIT=16*2**30


def array_bytes(value,seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value,np.ndarray):
        return value.nbytes
    if isinstance(value,dict):
        return sum(array_bytes(v,seen) for v in value.values())
    if isinstance(value,(list,tuple)):
        return sum(array_bytes(v,seen) for v in value)
    if hasattr(value,'__dict__'):
        return array_bytes(vars(value),seen)
    return 0


def experts_cached(checkpoint,trees,layer,group,arrays,proxy,beta,config,factor_sha):
    global _CACHE_BYTES
    result={}
    missing=[]
    for expert in map(int,group['experts']):
        key=(str(checkpoint),layer,expert,factor_sha)
        if key in _CACHE:
            result[expert]=_CACHE[key][0]
            _CACHE.move_to_end(key)
        else:
            missing.append(expert)
    if missing:
        cells=frozen.base._prepare_experts(SimpleNamespace(checkpoint=checkpoint,trees=trees,workers=4),layer,
            [dict(experts=missing)],arrays,proxy,beta,config)
        for expert,cell in cells.items():
            # Detach any NumPy views from large source arrays before retaining them.
            cell=copy.deepcopy(cell)
            size=array_bytes(cell)
            key=(str(checkpoint),layer,expert,factor_sha)
            while _CACHE and _CACHE_BYTES+size>_CACHE_LIMIT:
                _,(_,old_size)=_CACHE.popitem(last=False)
                _CACHE_BYTES-=old_size
            if size<=_CACHE_LIMIT:
                _CACHE[key]=(cell,size)
                _CACHE_BYTES+=size
            result[expert]=cell
    return result


def prepare(cap,layer,rate,checkpoint,trees,fit_dir,expected_factor):
    started=time.perf_counter()
    if hashlib.sha256(trees.read_bytes()).hexdigest()!=TREE_SHA:
        raise ValueError('tree changed')
    fp=fit_dir/f'average_rate_factor_layer_{layer}.npz'
    if hashlib.sha256(fp.read_bytes()).hexdigest()!=expected_factor:
        raise ValueError('live factor changed')
    selector,execution=frozen.selector_and_execution_router_weights(cap[f'scores_{layer}'],historical_sum_atol=5e-7,execution_sum_atol=.003)
    group=dict(group=0,activation=cap[f'x_{layer}'].reshape(-1),experts=cap[f'ids_{layer}'].reshape(-1),
               selector_weights=selector,execution_weights=execution)
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/qwen36_mxfp4_average_rate_all_layers.json').read_text())
    arrays=dict(np.load(fp,allow_pickle=False))
    proxy,beta=arrays['proxy'],float(arrays['beta'].reshape(-1)[0])
    with threadpool_limits(limits=1):
        experts=experts_cached(checkpoint,trees,layer,group,arrays,proxy,beta,config,expected_factor)
        coarse=frozen.base._group_geometry(group,experts,proxy,beta,config)
        geometry=frozen.RowCastGeometry(coarse['responses'],execution,proxy,beta)
        q4=sum(float(w)*np.asarray(response.target_output,np.float64) for w,response in zip(execution,coarse['responses']))
        error=float(np.max(np.abs(q4-cap[f'routed_{layer}'].reshape(-1))))
        if error>.125:
            raise RuntimeError(f'live Q4 reconstruction gate failed: {error}')
        frozen.base._REFINE_CONTEXT=dict(geometries=[coarse],groups=[group],config=config,rate=rate,proxy=proxy,beta=beta)
        _,refined=frozen.base._refine_geometry_task(0)
        states=frozen._pr13_and_local_states(refined,group,config,rate)
        pr13,local=states[:2]
        options=[]
        for kind,state in [('pr13',pr13),('exact_local',local)]:
            if any(np.array_equal(state,o['state']) for o in options):
                continue
            pages=frozen.state_page_count(state)
            if pages>8*rate:
                raise RuntimeError('live allocation exceeds budget')
            options.append(dict(kind=kind,state=state,delta=geometry.output_delta(state),pages=pages,
                                local_damage=geometry.local_damage(state)))
    frozen.base._REFINE_CONTEXT=None
    return options,dict(seconds=time.perf_counter()-started,q4_routed_max_abs=error,factor_sha256=expected_factor,
                        cpu_expert_cache_bytes=_CACHE_BYTES,cpu_expert_cache_limit_bytes=_CACHE_LIMIT)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--reference-bank',type=Path,required=True)
    p.add_argument('--layer',type=int,default=6)
    p.add_argument('--rate',type=int,default=360)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    bank=np.load(a.reference_bank)
    facts=json.loads(str(bank['facts_json']))
    options,timing=prepare(dict(np.load(a.capture)),a.layer,a.rate,Path('/workspace/qwen36_mxfp4_candidate'),
                          Path('/workspace/codebook_granularity_study/locked/selected_trees.json'),
                          Path('/workspace/pr13_average_rate_all_layers_20260822_v1/results/fit'),facts['factor_sha256'])
    rr=json.loads(str(bank['records_json']))
    for option in options:
        match=next(r for r in rr if r['rate']==a.rate and r['kind']==option['kind'])
        np.testing.assert_array_equal(option['state'],bank[match['key']+'_state'])
        np.testing.assert_array_equal(option['delta'],bank[match['key']+'_delta'])
    warm,warm_timing=prepare(dict(np.load(a.capture)),a.layer,a.rate,Path('/workspace/qwen36_mxfp4_candidate'),
                          Path('/workspace/codebook_granularity_study/locked/selected_trees.json'),
                          Path('/workspace/pr13_average_rate_all_layers_20260822_v1/results/fit'),facts['factor_sha256'])
    for before,after in zip(options,warm):
        np.testing.assert_array_equal(before['state'],after['state'])
        np.testing.assert_array_equal(before['delta'],after['delta'])
    timing['warm_seconds']=warm_timing['seconds']
    timing.update(options=len(options),states_and_deltas_bit_exact=True,rate=a.rate)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(timing,indent=2)+'\n')
    print(json.dumps(timing),flush=True)
