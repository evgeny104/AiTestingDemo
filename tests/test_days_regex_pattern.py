import re
import pytest


@pytest.mark.parametrize("text, should_match", [
    ("14 дней", True),
    ("140 дней", False),
    ("в 2014 году", False),
    ("Итог: 14", True)
])
def test_days_regex_matches_only_isolated_14(text, should_match):
    assert should_match == bool(
        re.search(r"\b14\b", text)), f"regex не сработал как ожидалось на строке {text!r} (ждали {should_match})"
