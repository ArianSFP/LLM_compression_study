import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("audit", Path(__file__).resolve().parents[1]/"scripts/audit_tail_value_cohort.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_full_prompt_uniqueness_does_not_hide_effective_overlap():
    rows = [dict(prompt_token_ids=[1,2,3,i], decode_position=2,
                 split=s, domain="d") for i,s in [(4,"calibration"),(5,"evaluation"),(6,"evaluation")]]
    facts = audit.audit(rows)
    assert facts["evaluation"]["requests"] == 2
    assert facts["evaluation"]["unique_effective_inputs"] == 1
    assert facts["evaluation_inputs_seen_in_calibration"] == 2


def test_corrected_split_excludes_old_prompts_and_is_deterministic():
    source = [dict(request_id=i,domain="d",prompt_sha256=str(i),
                   prompt_token_ids=[1]*40+[i]+[2]*40) for i in range(32)]
    old = [dict(r,decode_position=33,split="evaluation") for r in source[:16]]
    result = audit.corrected(source,old)
    assert result == audit.corrected(list(reversed(source)),old)
    assert len({audit.effective(r) for r in result}) == 16
    assert {s:sum(r["split"]==s for r in result) for s in ["development","validation","reserve"]} == dict(development=4,validation=4,reserve=8)
    assert not ({r["prompt_sha256"] for r in result}&{r["prompt_sha256"] for r in old})


def test_prefix_collision_fails_closed():
    source = [dict(request_id=i,domain="d",prompt_sha256=str(i),prompt_token_ids=[1]*70+[i]) for i in range(16)]
    with pytest.raises(ValueError,match="not unique"):
        audit.corrected(source,[])
