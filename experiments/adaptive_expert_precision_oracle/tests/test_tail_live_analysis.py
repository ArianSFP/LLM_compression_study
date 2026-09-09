from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_tail_live_decode import state_pages


@pytest.mark.parametrize('dtype',[np.int64,np.uint8])
def test_integer_state_storage_has_identical_page_count(dtype):
    state=np.zeros((8,512),dtype=dtype)
    state[0,:8]=np.arange(8)
    assert state_pages(state)==12


@pytest.mark.parametrize('invalid',[-1,8,1.5])
def test_invalid_state_values_are_rejected(invalid):
    state=np.zeros((8,512),dtype=np.float64 if invalid==1.5 else np.int64)
    state[0,0]=invalid
    with pytest.raises(AssertionError):
        state_pages(state)
