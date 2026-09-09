from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_tail_live_decode as live


def test_live_mapping_uses_current_layer_outputs_and_private_cache(monkeypatch):
    layers=[SimpleNamespace(mlp=SimpleNamespace(experts=torch.nn.Identity())) for _ in range(2)]
    model=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
    original_tensor=torch.tensor
    def cpu_tensor(*args,**kwargs):
        if kwargs.get('device')=='cuda':
            kwargs['device']='cpu'
        return original_tensor(*args,**kwargs)
    monkeypatch.setattr(live.torch,'tensor',cpu_tensor)
    def forward(model,tokens,cache,position):
        cache['seen']+=1
        value=original_tensor([[1.]],dtype=torch.bfloat16)
        for layer in layers:
            value=layer.mlp.experts(value)
        return SimpleNamespace(logits=value,past_key_values=cache)
    monkeypatch.setattr(live.base,'forward',forward)
    inherited={'seen':0}
    result=live.execute(model,1,0,inherited,{0:(np.array([[1.]]),np.array([1.])),1:(np.array([[2.]]),np.array([3.]))})
    assert float(result.logits.item())==5.
    assert inherited=={'seen':0}
    assert result.past_key_values=={'seen':1}
    with pytest.raises(RuntimeError,match='baseline changed'):
        live.execute(model,1,0,inherited,{1:(np.array([[999.]]),np.array([3.]))})
    # A failed trial must remove its hooks and leave the inherited state untouched.
    assert float(live.execute(model,1,0,inherited,{}).logits.item())==1.
    assert inherited=={'seen':0}
