#!/usr/bin/env python3
"""Descriptive within-bank headroom; never a runtime or promotion decision."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--predictions",type=Path)
    a=p.parse_args()
    predictions={}
    if a.predictions:
        pred=json.loads(a.predictions.read_text())
        if pred["terminal_labels_read"] is not False:
            raise ValueError("prediction provenance invalid")
        predictions={(r["request_id"],r["layer"],r["key"]):r["predicted_damage"] for r in pred["predictions"]}
    rows=[]
    for path in sorted((a.run/"labels").glob("*.json")):
        cell=json.loads(path.read_text())
        for r in cell["outcomes"]:
            rows.append(dict(**{k:cell[k] for k in ["request_id","domain","split","layer"]},**r))
    frame=pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no labels")
    contrasts=[]
    for (request,layer,rate),g in frame.groupby(["request_id","layer","rate"]):
        high=g[g.kind=="external_pr13_high"].iloc[0]
        low=g[g.kind=="pr13"].iloc[0]
        feasible=g[g.kind!="external_pr13_high"]
        restricted=feasible[feasible.d1_missing<=low.d1_missing]
        picks={"oracle":feasible.sort_values(["kl","key"]).iloc[0],
               "oracle_d1_nonworsening":restricted.sort_values(["kl","key"]).iloc[0],
               "local":feasible.sort_values(["local_damage","key"]).iloc[0],
               "d1_first":feasible.sort_values(["d1_missing","local_damage","key"]).iloc[0]}
        if predictions:
            feasible=feasible.copy()
            feasible["predicted_damage"]=[predictions[(request,int(layer),k)] for k in feasible.key]
            picks["metric"]=feasible.sort_values(["predicted_damage","key"]).iloc[0]
            picks["metric_d1_nonworsening"]=feasible[feasible.d1_missing<=low.d1_missing].sort_values(["predicted_damage","key"]).iloc[0]
        for policy,r in picks.items():
            contrasts.append(dict(request_id=request,domain=low.domain,split=low.split,layer=int(layer),rate=int(rate),
                     policy=policy,delta_vs_pr13_low=float(r.kl-low.kl),delta_vs_pr13_high=float(r.kl-high.kl),
                     delta_nll_vs_pr13_low=float(r.nll-low.nll),delta_nll_vs_pr13_high=float(r.nll-high.nll),
                     d1_constraint_regret=float(picks["oracle_d1_nonworsening"].kl-picks["oracle"].kl),
                     selected_kind=r.kind,pages=int(r.pages),reference_low_kl=float(low.kl)))
    contrast=pd.DataFrame(contrasts)
    req=contrast.groupby(["split","request_id","rate","policy"],as_index=False).mean(numeric_only=True)
    summaries=[]
    for (split,rate,policy),g in req.groupby(["split","rate","policy"]):
        s=dict(split=split,rate=int(rate),policy=policy,requests=len(g))
        for col in ["delta_vs_pr13_low","delta_vs_pr13_high","delta_nll_vs_pr13_low","delta_nll_vs_pr13_high","d1_constraint_regret"]:
            v=g[col].to_numpy()
            s[col]=dict(mean=float(v.mean()),median=float(np.median(v)),p95=float(np.quantile(v,.95)),
                        maximum=float(v.max()),negative_fraction=float(np.mean(v<0)),
                        harmed_fraction=float(np.mean(v>0)))
        summaries.append(s)
    for s in summaries:
        oracle=next(v for v in summaries if v["split"]==s["split"] and v["rate"]==s["rate"] and v["policy"]=="oracle")
        headroom=-oracle["delta_vs_pr13_low"]["mean"]
        s["fraction_same_payload_oracle_headroom_recovered"]=(-s["delta_vs_pr13_low"]["mean"]/headroom if headroom>1e-15 else None)
    strata=[]
    for keys,g in contrast.groupby(["split","rate","policy","layer","domain"]):
        split,rate,policy,layer,domain=keys
        strata.append(dict(split=split,rate=int(rate),policy=policy,layer=int(layer),domain=domain,
                           requests=g.request_id.nunique(),
                           mean_delta_vs_pr13_low=float(g.delta_vs_pr13_low.mean()),
                           mean_delta_vs_pr13_high=float(g.delta_vs_pr13_high.mean()),
                           mean_d1_constraint_regret=float(g.d1_constraint_regret.mean())))
    pd.DataFrame(strata).to_parquet(a.run/"stratum_contrasts.parquet",index=False)
    cache_rows=[]
    for path in sorted((a.run/"cache_probe").glob("*.json")):
        probe=json.loads(path.read_text())
        for rate in [360,725]:
            outcomes=[r for r in probe["outcomes"] if r["rate"]==rate]
            baseline={r["step"]:r for r in outcomes if r["policy"]=="pr13"}
            for policy in sorted({r["policy"] for r in outcomes}-{"pr13"}):
                candidate={r["step"]:r for r in outcomes if r["policy"]==policy}
                cache_rows.append(dict(request_id=probe["request_id"],domain=probe["domain"],rate=rate,policy=policy,
                    horizon=probe["horizon"],current_delta_kl=candidate[0]["kl"]-baseline[0]["kl"],
                    mean_future_delta_kl=float(np.mean([candidate[s]["kl"]-baseline[s]["kl"] for s in range(1,probe["horizon"]+1)]))))
    cache_summary=[]
    if cache_rows:
        cache_frame=pd.DataFrame(cache_rows)
        cache_frame.to_parquet(a.run/"cache_contrasts.parquet",index=False)
        for (rate,policy),g in cache_frame.groupby(["rate","policy"]):
            cache_summary.append(dict(rate=int(rate),policy=policy,requests=len(g),mean_current_delta_kl=float(g.current_delta_kl.mean()),
                mean_future_delta_kl=float(g.mean_future_delta_kl.mean()),
                future_harmed_fraction=float((g.mean_future_delta_kl>0).mean()),
                worst_request_mean_future_delta_kl=float(g.mean_future_delta_kl.max())))
    frame.to_parquet(a.run/"candidate_labels.parquet",index=False)
    contrast.to_parquet(a.run/"cell_contrasts.parquet",index=False)
    req.to_parquet(a.run/"request_contrasts.parquet",index=False)
    facts=dict(schema="tail_value_descriptive_v1",candidate_rows=len(frame),cells=frame.groupby(["request_id","layer","rate"]).ngroups,
               completed=(a.run/"completed.json").exists(),summaries=summaries,
               inference="exploratory development/validation; bank optimum is hindsight; no Experiment B promotion; extra learned metadata unpriced",
               predictions_model_sha256=(pred["model_sha256"] if predictions else None),cache_summary=cache_summary)
    (a.run/"analysis.json").write_text(json.dumps(facts,indent=2)+"\n")
    print(json.dumps(facts,indent=2))


if __name__=="__main__":
    main()
