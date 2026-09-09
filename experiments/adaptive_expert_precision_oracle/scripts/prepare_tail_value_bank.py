#!/usr/bin/env python3
"""CPU-only frozen candidate generation; never reads terminal labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_d1_slice_oracle_pilot as base
from run_d1_nested_safe_oracle import _pr13_and_local_states
from run_same_host_causal_controls import selector_and_execution_router_weights
from oracle_study.d1_nested_geometry import TokenStateGeometry
from oracle_study.split_interaction_field import SPLIT_STATE_HIDDEN_INDEX, SPLIT_STATE_DOWN_HIGH, split_state_output
from oracle_study.d1_nested_allocation import (state_page_count, legal_add_moves, legal_remove_moves, apply_page_move,
    reverse_local_prune_to_reserve, add_only_local_completion, validate_states)


class RowCastGeometry(TokenStateGeometry):
    """Same arithmetic as TokenStateGeometry, without casting a whole matrix per move."""
    def _unit_output(self, expert, unit, state):
        response=self.responses[int(expert)]
        hidden=float(response.hidden[int(unit),int(SPLIT_STATE_HIDDEN_INDEX[int(state)])])
        matrix=response.down4 if bool(SPLIT_STATE_DOWN_HIGH[int(state)]) else response.down2
        return hidden*np.asarray(matrix[int(unit)],np.float64)

    def output_delta(self, states):
        """Cache unchanged expert rows, retaining the original sum order exactly."""
        value=validate_states(states)
        key=np.asarray(value,np.uint8).tobytes(order="C")
        if key in self._output_delta_cache:
            self._output_delta_cache_hits[0]+=1
            return self._output_delta_cache[key]
        if not hasattr(self,"_weighted_rows"):
            object.__setattr__(self,"_weighted_rows",{})
        result=np.zeros(self.output_width,np.float64)
        for expert,response in enumerate(self.responses):
            rowkey=(expert,value[expert].tobytes())
            if rowkey not in self._weighted_rows:
                approximate=split_state_output(response,value[expert])
                self._weighted_rows[rowkey]=float(self.execution_weights[expert])*(
                    np.asarray(approximate,np.float64)-np.asarray(response.target_output,np.float64))
            result+=self._weighted_rows[rowkey]
        result.setflags(write=False)
        self._output_delta_cache[key]=result
        return result

    def move_output_deltas(self, states, moves):
        """Vectorize exact unit effects while preserving each arithmetic operation."""
        state=validate_states(states)
        if not moves:
            return np.empty((0,self.output_width),np.float64)
        table=np.asarray([(m.expert,m.unit,m.source_state,m.destination_state) for m in moves],np.int64)
        if np.any(state[table[:,0],table[:,1]]!=table[:,2]):
            raise ValueError("page move sequence contains a stale move")
        result=np.empty((len(moves),self.output_width),np.float64)
        for expert in np.unique(table[:,0]):
            ix=np.flatnonzero(table[:,0]==expert)
            units,source,destination=table[ix,1],table[ix,2],table[ix,3]
            response=self.responses[int(expert)]
            d2=np.asarray(response.down2[units],np.float64)
            d4=np.asarray(response.down4[units],np.float64)
            hidden=np.asarray(response.hidden,np.float64)
            before=hidden[units,SPLIT_STATE_HIDDEN_INDEX[source],None]*np.where(SPLIT_STATE_DOWN_HIGH[source,None],d4,d2)
            after=hidden[units,SPLIT_STATE_HIDDEN_INDEX[destination],None]*np.where(SPLIT_STATE_DOWN_HIGH[destination,None],d4,d2)
            result[ix]=float(self.execution_weights[int(expert)])*(after-before)
        return result


def candidates(pr13, local, high, ids, geometry, low_rate, high_rate):
    # The old helper also builds an unused high completion. Omitting that work
    # leaves core/low states unchanged; the external high remains original PR13.
    core = reverse_local_prune_to_reserve(high, ids, budget_pages=8*low_rate,
        repair_window_pages=2, local_damage=geometry.local_damage,
        score_removals=geometry.score_moves, refresh_after_accepted_pages=1).final.states
    low = add_only_local_completion(core, ids, budget_pages=8*low_rate,
        local_damage=geometry.local_damage, score_additions=geometry.score_moves,
        refresh_after_accepted_pages=1)
    rows = [("pr13", pr13), ("exact_local", local), ("nested_local", low.final.states), ("core", core)]
    adds = legal_add_moves(core)
    scores = geometry.score_moves(core, adds)
    order = np.argsort(scores, kind="stable")
    for j in order[:4]:
        rows.append(("single", apply_page_move(core, adds[j])))
    pairs = []
    for j in order[:8]:
        first = apply_page_move(core, adds[j])
        second_moves = legal_add_moves(first)
        second_order = np.argsort(geometry.score_moves(first, second_moves), kind="stable")
        pairs.append(apply_page_move(first, second_moves[second_order[0]]))
    # Include same-unit gate/up combinations regardless of single-page ranking.
    for expert, unit in np.argwhere((core & 6) == 0):
        state = core.copy()
        state[expert, unit] |= 6
        pairs.append(state)
    pairs.sort(key=geometry.local_damage)
    rows += [("pair", s) for s in pairs[:4]]
    removes = legal_remove_moves(pr13)
    for j in np.argsort(geometry.score_moves(pr13, removes), kind="stable")[:4]:
        removed = apply_page_move(pr13, removes[j])
        moves = legal_add_moves(removed)
        for k in np.argsort(geometry.score_moves(removed, moves), kind="stable"):
            swapped = apply_page_move(removed, moves[k])
            if not np.array_equal(swapped, pr13):
                rows.append(("swap", swapped))
                break
    unique = {}
    for name, state in rows:
        if state_page_count(state) > 8*low_rate:
            raise RuntimeError("candidate exceeds cap")
        unique.setdefault(state.tobytes(), (name, state))
    return list(unique.values()) + [("external_pr13_high", high)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("/workspace/qwen36_mxfp4_candidate"))
    p.add_argument("--trees", type=Path, default=Path("/workspace/codebook_granularity_study/locked/selected_trees.json"))
    p.add_argument("--fit-dir", type=Path, default=Path("/workspace/pr13_average_rate_all_layers_20260822_v1/results/fit"))
    a = p.parse_args()
    start = time.perf_counter()
    if hashlib.sha256(a.trees.read_bytes()).hexdigest()!="da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8":
        raise ValueError("selected quantizer tree identity differs")
    cap = np.load(a.capture, allow_pickle=False)
    layer = a.layer
    selector, execution = selector_and_execution_router_weights(cap[f"scores_{layer}"], historical_sum_atol=5e-7, execution_sum_atol=.003)
    group = dict(group=0, activation=cap[f"x_{layer}"].reshape(-1), experts=cap[f"ids_{layer}"].reshape(-1),
                 selector_weights=selector, execution_weights=execution)
    config = json.loads((Path(__file__).resolve().parents[1]/"configs/qwen36_mxfp4_average_rate_all_layers.json").read_text())
    arrays = dict(np.load(a.fit_dir/f"average_rate_factor_layer_{layer}.npz", allow_pickle=False))
    proxy, beta = arrays["proxy"], float(arrays["beta"].reshape(-1)[0])
    with threadpool_limits(limits=1):
        experts = base._prepare_experts(SimpleNamespace(checkpoint=a.checkpoint, trees=a.trees, workers=4), layer, [group], arrays, proxy, beta, config)
        coarse = base._group_geometry(group, experts, proxy, beta, config)
        print("coarse geometry completed", time.perf_counter()-start, flush=True)
        geometry = RowCastGeometry(coarse["responses"], execution, proxy, beta)
        q4 = sum(float(w)*np.asarray(response.target_output,np.float64)
                 for w,response in zip(execution,coarse["responses"]))
        q4_error = float(np.max(np.abs(q4-cap[f"routed_{layer}"].reshape(-1))))
        if q4_error > 0.125:
            raise RuntimeError(f"Q4 reconstruction parity exceeded original 0.125 tolerance: {q4_error}")
        states = {}
        for rate in [360,384,725,749]:
            base._REFINE_CONTEXT = dict(geometries=[coarse],groups=[group],config=config,rate=rate,proxy=proxy,beta=beta)
            _, refined = base._refine_geometry_task(0)
            states[rate] = _pr13_and_local_states(refined, group, config, rate)
            print("rate completed", rate, time.perf_counter()-start, flush=True)
        values, facts = {}, []
        for low, high in [(360,384),(725,749)]:
            bank = candidates(states[low][0], states[low][1], states[high][0], group["experts"], geometry,low,high)
            realized = {}
            import torch
            baseline = torch.tensor(cap[f"routed_{layer}"].reshape(-1)).to(torch.bfloat16)
            for i, (kind, state) in enumerate(bank):
                key=f"r{low}_c{i}"
                delta=geometry.output_delta(state)
                injected=(baseline.float()+torch.tensor(delta,dtype=torch.float32)).to(torch.bfloat16)
                eh=hashlib.sha256(injected.view(torch.uint16).numpy().tobytes()).hexdigest()
                record=dict(key=key,rate=low,kind=kind,pages=state_page_count(state),local_damage=geometry.local_damage(state),
                            state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),execution_sha256=eh,
                            duplicate_execution_of=realized.get(eh))
                realized.setdefault(eh,key)
                values[key+"_state"]=state
                values[key+"_delta"]=delta
                facts.append(record)
        values["records_json"] = np.asarray(json.dumps(facts))
        values["facts_json"] = np.asarray(json.dumps(dict(schema="tail_value_bank_v1",layer=layer,
                    capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(), seconds=time.perf_counter()-start,
                    q4_routed_max_abs=q4_error, q4_routed_max_abs_tolerance=0.125,
                    terminal_labels_read=False, factor_sha256=hashlib.sha256((a.fit_dir/f"average_rate_factor_layer_{layer}.npz").read_bytes()).hexdigest())))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        tmp=a.output.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f,**values)
        tmp.replace(a.output)
        print(json.dumps(dict(bank=str(a.output), candidates=len(facts),seconds=time.perf_counter()-start)),flush=True)


if __name__ == "__main__":
    main()
