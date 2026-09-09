#!/usr/bin/env python3
"""Offline frozen-bank tail labels on the 3090, with atomic per-cell results."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oracle_study.d1_decode import cache_state_metrics
from tail_value_streaming import load_streamed, forward, expert_cache_stats


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def record_observers(model, layers):
    data, handles = {}, []
    for i, layer in enumerate(model.model.language_model.layers):
        def gate_hook(m, args, out, i=i):
            data[f"ids_{i}"] = out[2].detach().cpu().clone()
            data[f"scores_{i}"] = out[1].detach().float().cpu().clone()
        handles.append(layer.mlp.gate.register_forward_hook(gate_hook))
        if i in layers:
            def expert_hook(m, args, out, i=i):
                data[f"x_{i}"] = args[0].detach().float().cpu().clone()
                data[f"routed_{i}"] = out.detach().float().cpu().clone()
            handles.append(layer.mlp.experts.register_forward_hook(expert_hook))
    return data, handles


def quality(reference, candidate, target):
    logp = reference.float().reshape(-1).log_softmax(-1)
    logq = candidate.float().reshape(-1).log_softmax(-1)
    kl = float((logp.exp() * (logp-logq)).sum())
    return dict(kl=kl, nll=float(-logq[target]), delta_nll=float(logp[target]-logq[target]))


def cache_probe(model, row, prefix, ref, baseline, bank, outcomes, output):
    """One layer-6 injection followed by exact decode on each private cache."""
    tokens, pos=row["prompt_token_ids"],row["decode_position"]
    horizon=min(4,len(tokens)-pos-2)
    if horizon<1:
        return
    refs=[ref.logits.detach().clone()]
    cache=copy.deepcopy(ref.past_key_values)
    for step in range(1,horizon+1):
        out=forward(model,[tokens[pos+step]],cache,pos+step)
        cache=out.past_key_values
        refs.append(out.logits.detach().clone())
    del out,cache
    result=[]
    for rate in [360,725]:
        feasible=[r for r in outcomes if r["rate"]==rate and r["kind"]!="external_pr13_high"]
        choices={"pr13":next(r for r in feasible if r["kind"]=="pr13"),
                 "current_token_oracle":min(feasible,key=lambda r:(r["kl"],r["key"]))}
        for policy,record in choices.items():
            delta=torch.tensor(bank[record["key"]+"_delta"],device="cuda",dtype=torch.float32)
            def inject(m,args,out):
                if not torch.equal(out,baseline):
                    raise RuntimeError("cache probe injection baseline differs")
                return (out.float()+delta).to(out.dtype)
            h=model.model.language_model.layers[6].mlp.experts.register_forward_hook(inject)
            try:
                candidate=forward(model,[tokens[pos]],copy.deepcopy(prefix),pos)
            finally:
                h.remove()
            for step in range(horizon+1):
                if step:
                    candidate=forward(model,[tokens[pos+step]],candidate.past_key_values,pos+step)
                result.append(dict(rate=rate,policy=policy,step=step,selected_key=record["key"],
                                   **quality(refs[step],candidate.logits,tokens[pos+step+1])))
            del candidate
    atomic_json(output,dict(request_id=row["request_id"],domain=row["domain"],horizon=horizon,
                           mode="single_layer_single_injection_private_cache_then_exact_decode",outcomes=result))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/workspace/qwen36_mxfp4_candidate")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-domain", type=int, default=4)
    p.add_argument("--layers", default="0,1,4,6,12,23")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    rows = [json.loads(l) for l in a.manifest.read_text().splitlines()]
    if any(r["split"] == "reserve" for r in rows):
        raise ValueError("reserved requests may not be evaluated")
    rows = [r for d in sorted({r["domain"] for r in rows}) for r in [r for r in rows if r["domain"] == d][:a.per_domain]]
    layers = list(map(int, a.layers.split(",")))
    a.output.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    identity = dict(schema="tail_value_run_v1",manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                    request_ids=[r["request_id"] for r in rows],layers=layers,
                    code_sha256={name:hashlib.sha256((script_dir/name).read_bytes()).hexdigest() for name in
                                 ["run_tail_value_3090.py","prepare_tail_value_bank.py","tail_value_streaming.py"]})
    identity_path=a.output/"identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("resume identity changed")
    atomic_json(identity_path,identity)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter()
    model=load_streamed(a.checkpoint)
    capture, handles=record_observers(model,layers)
    # Materialize small current-token captures; prefix caches are rebuilt for labels.
    for r in rows:
        path=a.output/"captures"/(r["request_id"]+".npz")
        if path.exists():
            continue
        tokens,position=r["prompt_token_ids"],r["decode_position"]
        pref=forward(model,tokens[:position])
        ref=forward(model,[tokens[position]],pref.past_key_values,position)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(f,**{k:v.numpy() for k,v in capture.items()})
        path.with_suffix(".tmp").replace(path)
        del ref,pref
        print("captured",r["domain"],r["request_id"],flush=True)

    def prepare(job):
        r,layer=job
        path=a.output/"banks"/f"{r['request_id']}_l{layer}.npz"
        if path.exists():
            return
        log=a.output/"logs"/f"bank_{r['request_id']}_l{layer}.log"
        log.parent.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        with log.open("w") as f:
            subprocess.run([sys.executable,"-u",str(script_dir/"prepare_tail_value_bank.py"),
                            "--capture",str(a.output/"captures"/(r["request_id"]+".npz")),
                            "--layer",str(layer),"--output",str(path),"--checkpoint",a.checkpoint],
                           stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print("bank sealed",r["request_id"],layer,flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(prepare,[(r,l) for r in rows for l in layers]))
    bank_seal={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((a.output/"banks").glob("*.npz"))}
    atomic_json(a.output/"bank_seal.json",bank_seal)

    for r in rows:
        tokens,position=r["prompt_token_ids"],r["decode_position"]
        if all((a.output/"labels"/f"{r['request_id']}_l{l}.json").exists() for l in layers):
            continue
        pref=forward(model,tokens[:position])
        prefix=pref.past_key_values
        del pref
        ref=forward(model,[tokens[position]],copy.deepcopy(prefix),position)
        base_capture={k:v.clone() for k,v in capture.items()}
        repeated=forward(model,[tokens[position]],copy.deepcopy(prefix),position)
        zero=cache_state_metrics(ref.past_key_values,repeated.past_key_values).to_dict()
        if not torch.equal(ref.logits,repeated.logits) or not zero["cache_bit_identical"]:
            raise RuntimeError("same-host zero-dose failed")
        del repeated
        stored=np.load(a.output/"captures"/(r["request_id"]+".npz"))
        if any(not np.array_equal(stored[k],v.numpy()) for k,v in base_capture.items()):
            raise RuntimeError("allocation capture differs from label reference")
        atomic_json(a.output/"zero_gates"/(r["request_id"]+".json"),dict(logits_bit_exact=True,**zero))
        for layer in layers:
            outpath=a.output/"labels"/f"{r['request_id']}_l{layer}.json"
            if outpath.exists():
                continue
            bp=a.output/"banks"/f"{r['request_id']}_l{layer}.npz"
            if hashlib.sha256(bp.read_bytes()).hexdigest()!=bank_seal[bp.name]:
                raise RuntimeError("sealed bank changed")
            bank=np.load(bp)
            records=json.loads(str(bank["records_json"]))
            outcomes, memo=[],{}
            for record in records:
                execution=record["execution_sha256"]
                if execution not in memo:
                    delta=torch.tensor(bank[record["key"]+"_delta"],device="cuda",dtype=torch.float32)
                    baseline=base_capture[f"routed_{layer}"].to("cuda",dtype=torch.bfloat16)
                    def inject(m,args,out):
                        if not torch.equal(out,baseline):
                            raise RuntimeError("injection baseline differs")
                        return (out.float()+delta).to(out.dtype)
                    handle=model.model.language_model.layers[layer].mlp.experts.register_forward_hook(inject)
                    t=time.perf_counter()
                    try:
                        candidate=forward(model,[tokens[position]],copy.deepcopy(prefix),position)
                    finally:
                        handle.remove()
                    ids=set(capture[f"ids_{layer+1}"].reshape(-1).tolist())
                    reference_ids=set(base_capture[f"ids_{layer+1}"].reshape(-1).tolist())
                    memo[execution]=dict(**quality(ref.logits,candidate.logits,tokens[position+1]),
                                         d1_missing=len(reference_ids-ids),seconds=time.perf_counter()-t)
                    del candidate
                outcomes.append(dict(record,**memo[execution]))
            atomic_json(outpath,dict(request_id=r["request_id"],domain=r["domain"],split=r["split"],layer=layer,
                                     position=position,bank_sha256=bank_seal[bp.name],outcomes=outcomes))
            print("labelled",r["domain"],r["request_id"],layer,len(outcomes),flush=True)
            if layer==6 and r==next(row for row in rows if row["domain"]==r["domain"]):
                cache_probe(model,r,prefix,ref,base_capture["routed_6"].to("cuda",dtype=torch.bfloat16),
                            bank,outcomes,a.output/"cache_probe"/(r["request_id"]+".json"))
        del ref,prefix
    atomic_json(a.output/"completed.json",dict(completed=True,requests=len(rows),layers=layers,
                elapsed_seconds=time.perf_counter()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                expert_weight_cache=expert_cache_stats()))


if __name__ == "__main__":
    main()
