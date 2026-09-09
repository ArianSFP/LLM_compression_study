#!/usr/bin/env python3
"""Small PSD metric, trained on development pairs; prediction reads no labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls


def features(bank, records, proxy):
    return np.asarray([[r['local_damage'], *np.square(np.asarray(bank[r['key']+'_delta']) @ proxy)]
                       for r in records],dtype=np.float64)


def fit_pairs(x, y, baseline):
    """Return a positive local floor plus nonnegative paired-fit correction."""
    dx=x-x[baseline]
    dy=y-y[baseline]
    floor=.05*max(float(x[:,0]@y/max(float(x[:,0]@x[:,0]),1e-30)),1e-12)
    scale=np.maximum(np.sqrt(np.mean(dx*dx,axis=0)),1e-20)
    target=dy-floor*dx[:,0]
    n=len(y)
    coef,_=nnls(np.concatenate([dx/scale,np.sqrt(n*.001)*np.eye(x.shape[1])]),
                np.concatenate([target,np.zeros(x.shape[1])]))
    coef=coef/scale
    coef[0]+=floor
    return coef,dict(local_floor=floor,ridge=.001,pairs=n,
                     pair_rmse=float(np.sqrt(np.mean((dx@coef-dy)**2))))


def load_banks(run, fit_dir):
    seal=json.loads((run/'bank_seal.json').read_text())
    rows=[]
    for name,sha in sorted(seal.items()):
        path=run/'banks'/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('bank seal mismatch')
        with np.load(path) as bank:
            facts=json.loads(str(bank['facts_json']))
            layer=facts['layer']
            factor=fit_dir/f'average_rate_factor_layer_{layer}.npz'
            if hashlib.sha256(factor.read_bytes()).hexdigest()!=facts['factor_sha256']:
                raise ValueError('factor identity changed')
            proxy=np.load(factor)['proxy']
            records=json.loads(str(bank['records_json']))
            xs=features(bank,records,proxy)
            request=name.rsplit('_l',1)[0]
            for r,x in zip(records,xs):
                if r['kind']!='external_pr13_high':
                    rows.append(dict(request_id=request,layer=layer,features=x.tolist(),**r))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['fit','predict'])
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--fit-dir',type=Path,default=Path('/workspace/pr13_average_rate_all_layers_20260822_v1/results/fit'))
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    rows=load_banks(a.run,a.fit_dir)
    if a.mode=='fit':
        labels={}
        for path in (a.run/'labels').glob('*.json'):
            c=json.loads(path.read_text())
            if c['split']!='development':
                raise ValueError('fit accepts development only')
            for r in c['outcomes']:
                labels[(c['request_id'],c['layer'],r['key'])]=r['kl']
        if not (a.run/'completed.json').exists():
            raise ValueError('development incomplete')
        fitted={}
        for layer in sorted({r['layer'] for r in rows}):
            rr=[r for r in rows if r['layer']==layer]
            x=np.asarray([r['features'] for r in rr])
            y=np.asarray([labels[(r['request_id'],layer,r['key'])] for r in rr])
            bases={(r['request_id'],r['rate']):i for i,r in enumerate(rr) if r['kind']=='pr13'}
            baseline=np.asarray([bases[(r['request_id'],r['rate'])] for r in rr])
            coef,diagnostic=fit_pairs(x,y,baseline)
            fitted[str(layer)]=dict(coefficients=coef.tolist(),**diagnostic)
        model=dict(schema='tail_value_psd_proxy4_v1',feature_names=['local_damage','proxy0_squared','proxy1_squared','proxy2_squared','proxy3_squared'],
                   training_bank_seal_sha256=hashlib.sha256((a.run/'bank_seal.json').read_bytes()).hexdigest(),
                   features_require_exact_q4_residual=True,layers=fitted)
        if a.model.exists():
            raise ValueError('refusing to overwrite frozen model')
        a.model.write_text(json.dumps(model,indent=2)+'\n')
        print(json.dumps(model,indent=2))
    else:
        if a.output is None:
            raise ValueError('--output required')
        model=json.loads(a.model.read_text())
        predictions=[dict(request_id=r['request_id'],layer=r['layer'],rate=r['rate'],key=r['key'],
                          predicted_damage=float(np.asarray(r['features'])@model['layers'][str(r['layer'])]['coefficients'])) for r in rows]
        output=dict(model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),terminal_labels_read=False,predictions=predictions)
        a.output.write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps(dict(predictions=len(predictions),terminal_labels_read=False)))

if __name__=='__main__':
    main()
