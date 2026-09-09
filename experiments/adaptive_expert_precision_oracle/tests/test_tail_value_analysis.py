import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_tail_value_bank as analysis


def test_d1_opportunity_cost_and_future_reversal_are_reported(tmp_path,monkeypatch):
    (tmp_path/'labels').mkdir()
    (tmp_path/'cache_probe').mkdir()
    rows=[]
    cache=[]
    for rate in [360,725]:
        for kind,kl,d1,damage in [('pr13',2.,0,2.),('pair',1.,1,3.),('external_pr13_high',.5,0,1.)]:
            rows.append(dict(key=f'{rate}_{kind}',rate=rate,kind=kind,kl=kl,nll=kl+1.,d1_missing=d1,local_damage=damage,pages=8*rate))
        for policy in ['pr13','current_token_oracle']:
            for step in [0,1]:
                kl=2. if policy=='pr13' else (1. if step==0 else 3.)
                cache.append(dict(rate=rate,policy=policy,step=step,kl=kl))
    (tmp_path/'labels'/'request_l6.json').write_text(json.dumps(dict(request_id='request',domain='example',split='development',layer=6,outcomes=rows)))
    (tmp_path/'cache_probe'/'request.json').write_text(json.dumps(dict(request_id='request',domain='example',horizon=1,outcomes=cache)))
    monkeypatch.setattr(sys,'argv',['analyze','--run',str(tmp_path)])
    analysis.main()
    result=json.loads((tmp_path/'analysis.json').read_text())
    oracle=next(r for r in result['summaries'] if r['rate']==360 and r['policy']=='oracle')
    restricted=next(r for r in result['summaries'] if r['rate']==360 and r['policy']=='oracle_d1_nonworsening')
    assert oracle['delta_vs_pr13_low']['mean']==-1.
    assert restricted['delta_vs_pr13_low']['mean']==0.
    assert restricted['d1_constraint_regret']['mean']==1.
    assert result['cache_summary'][0]['mean_current_delta_kl']==-1.
    assert result['cache_summary'][0]['mean_future_delta_kl']==1.
    assert result['cache_summary'][0]['future_harmed_fraction']==1.
