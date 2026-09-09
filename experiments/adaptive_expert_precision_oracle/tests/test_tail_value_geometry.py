from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
from prepare_tail_value_bank import RowCastGeometry
from oracle_study.d1_nested_geometry import TokenStateGeometry
from oracle_study.d1_nested_allocation import legal_add_moves,legal_remove_moves
from oracle_study.split_interaction_field import SplitProjectionResponses


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_vectorized_effects_and_scores_are_bit_identical(dtype):
    rng=np.random.default_rng(481)
    responses=tuple(SplitProjectionResponses(rng.normal(size=(512,4)).astype(dtype),
        rng.normal(size=(512,17)).astype(dtype),rng.normal(size=(512,17)).astype(dtype)) for _ in range(8))
    weights=rng.uniform(size=8);weights/=weights.sum()
    proxy=rng.normal(size=(17,4))
    original=TokenStateGeometry(responses,weights,proxy,.7)
    fast=RowCastGeometry(responses,weights,proxy,.7)
    states=rng.integers(0,8,size=(8,512),dtype=np.uint8)
    moves=(legal_add_moves(states)+legal_remove_moves(states))[::23]
    np.testing.assert_array_equal(original.move_output_deltas(states,moves),fast.move_output_deltas(states,moves))
    np.testing.assert_array_equal(original.score_moves(states,moves),fast.score_moves(states,moves))
    for i in range(20):
        changed=states.copy()
        changed[i%8,i%512]^=1
        np.testing.assert_array_equal(original.output_delta(changed),fast.output_delta(changed))
        assert original.local_damage(changed)==fast.local_damage(changed)
