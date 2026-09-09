#!/usr/bin/env python3
"""Optimistic feature-capacity diagnostic; this is not a learned selector."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fit_tail_value_metric import load_banks


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--fit-dir',type=Path,required=True)
    a=p.parse_args()
    rows=load_banks(a.run,a.fit_dir)
    labels={}
    for path in (a.run/'labels').glob('*.json'):
        c=json.loads(path.read_text())
        for r in c['outcomes']:
            labels[(c['request_id'],c['layer'],r['key'])]=r['kl']
    frame=pd.DataFrame(rows)
    cells=[]
    for (request,layer,rate),g in frame.groupby(['request_id','layer','rate']):
        rr=g.to_dict('records')
        x=np.asarray([r['features'] for r in rr])
        kl=np.asarray([labels[(request,int(layer),r['key'])] for r in rr])
        tolerance=max(1e-15,1e-10*float(np.max(np.abs(x[:,0]))))
        dominated=np.asarray([bool(np.any((x[:,0]<x[j,0]-tolerance)&np.all(x[:,1:]<=x[j,1:],axis=1))) for j in range(len(x))])
        baseline=next(i for i,r in enumerate(rr) if r['kind']=='pr13')
        best=min(range(len(rr)),key=lambda i:(kl[i],rr[i]['key']))
        envelope=min(float(kl[~dominated].min()),float(kl[baseline]))
        cells.append(dict(request_id=request,layer=int(layer),rate=int(rate),oracle_winner_dominated=bool(dominated[best]),
                          dominated_candidates=int(dominated.sum()),candidates=len(rr),pr13_kl=float(kl[baseline]),
                          oracle_kl=float(kl[best]),optimistic_nondominated_or_incumbent_kl=envelope))
    f=pd.DataFrame(cells)
    f.to_parquet(a.run/'metric_dominance_cells.parquet',index=False)
    summary=[]
    for rate,g in f.groupby('rate'):
        full=float((g.pr13_kl-g.oracle_kl).mean())
        envelope=float((g.pr13_kl-g.optimistic_nondominated_or_incumbent_kl).mean())
        summary.append(dict(rate=int(rate),cells=len(g),oracle_winner_dominated_fraction=float(g.oracle_winner_dominated.mean()),
            fraction_oracle_headroom_remaining_in_optimistic_envelope=envelope/full if full>0 else None))
    result=dict(schema='tail_metric_dominance_v1',summaries=summary,
        interpretation='Strictly lower local damage and no larger squared proxy feature dominates under positive-local/nonnegative-proxy metrics. The nondominated set plus incumbent is an optimistic capacity envelope, selected with terminal labels; it is not a realizable trained policy or a global allocation bound.')
    (a.run/'metric_dominance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
