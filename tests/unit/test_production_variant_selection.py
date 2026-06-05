from __future__ import annotations

import importlib.util

from src import config


def _load_script_module(script_name: str, module_name: str):
    script_path = config.REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selector_picks_variant_with_highest_score() -> None:
    module = _load_script_module("21_select_production_variant.py", "select_variant_script")
    selected, tie_break_used, delta_pp = module.select_variant_by_score(
        scores={"A": 0.91, "B": 0.88, "C": 0.84},
        median_input_tokens={"A": 8200.0, "B": 8100.0, "C": 7900.0},
    )
    assert selected == "A"
    assert tie_break_used is False
    assert delta_pp > 1.0


def test_selector_applies_tie_break_by_lower_input_tokens_within_1pp() -> None:
    module = _load_script_module("21_select_production_variant.py", "select_variant_script_tie")
    selected, tie_break_used, delta_pp = module.select_variant_by_score(
        scores={"A": 0.905, "B": 0.899, "C": 0.830},
        median_input_tokens={"A": 8600.0, "B": 8200.0, "C": 7800.0},
    )
    assert selected == "B"
    assert tie_break_used is True
    assert delta_pp <= 1.0

