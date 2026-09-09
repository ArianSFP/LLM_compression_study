from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_tail_value_metric import fit_pairs


def test_psd_pair_fit_recovers_known_metric_and_preserves_positive_floor():
    rng=np.random.default_rng(38)
    x=rng.lognormal(size=(200,5))
    truth=np.array([.3,.1,.8,.0,.2])
    y=x@truth
    baseline=np.repeat(np.arange(0,200,10),10)
    fitted,facts=fit_pairs(x,y,baseline)
    assert fitted[0]>=facts['local_floor']>0
    assert np.all(fitted>=0)
    gain=(x-x[baseline])@truth
    predicted=(x-x[baseline])@fitted
    assert np.linalg.norm(predicted-gain)/np.linalg.norm(gain)<.01


def test_psd_fit_cannot_learn_negative_quadratic_cost():
    rng=np.random.default_rng(46)
    x=rng.uniform(.1,1,size=(100,5))
    y=10-x[:,1]
    fitted,facts=fit_pairs(x,y,np.zeros(100,dtype=int))
    assert fitted[0]>=facts['local_floor']>0
    assert np.all(fitted>=0)
    assert np.all(x@fitted>0)
