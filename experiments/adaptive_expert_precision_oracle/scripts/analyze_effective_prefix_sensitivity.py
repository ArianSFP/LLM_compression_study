#!/usr/bin/env python3
"""Reproduce descriptive cohort-weighting sensitivity; preserve original decision."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-analysis",type=Path,required=True)
    p.add_argument("--membership",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    membership=pd.DataFrame(json.loads(a.membership.read_text()))
    requests=pd.read_parquet(a.original_analysis/"d1_nested_primary_request_differences.parquet")
    layers=pd.read_parquet(a.original_analysis/"d1_nested_primary_request_layer_differences.parquet")
    merged=requests.merge(membership[["request_id","effective_input_sha256"]],on="request_id",validate="many_to_one")
    rows=[]
    for name,g in merged.groupby("contrast"):
        spread=g.groupby("effective_input_sha256").logit_kl_difference.agg(["min","max"])
        if not (spread["min"]==spread["max"]).all():
            raise ValueError("identical effective inputs have different outcomes")
        rows.append(dict(contrast=name,requests=len(g),distinct_effective_inputs=len(spread),
                         original_mean=float(g.logit_kl_difference.mean()),
                         equal_distinct_input_mean=float(spread["min"].mean())))
    sentinel=layers[(layers.injection_layer==6)&(layers.contrast=="nested_d1_safe_360_vs_independent_pr13_384_product_matched")]
    summary=sentinel[sentinel.domain=="summarisation"]
    facts=dict(schema="effective_prefix_sensitivity_v1",contrasts=rows,
        summarisation_fraction_of_sentinel_net_regression=float(summary.logit_kl_difference.sum()/sentinel.logit_kl_difference.sum()),
        original_promotion_decision="failed_unchanged",
        interpretation="Equal distinct-input weighting changes domain weights and the estimand. Descriptive sensitivity only, not corrected confirmatory inference or a promotion.")
    a.output.write_text(json.dumps(facts,indent=2)+"\n")
    print(json.dumps(facts,indent=2))


if __name__=="__main__":
    main()
