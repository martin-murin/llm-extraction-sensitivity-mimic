from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "26_pipeline_diagnostics_5k.py"
    spec = importlib.util.spec_from_file_location("pipeline_diagnostics_5k", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/26_pipeline_diagnostics_5k.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_per_variant_cost_baseline_lookup() -> None:
    mod = _load_module()
    passed_a, msg_a = mod.cost_within_5pct("a", 0.001901)
    passed_b, msg_b = mod.cost_within_5pct("b", 0.002001)
    passed_c, msg_c = mod.cost_within_5pct("c", 0.002060)
    assert passed_a and "vs variant A 1k baseline" in msg_a
    assert passed_b and "vs variant B 1k baseline" in msg_b
    assert passed_c and "vs variant C 1k baseline" in msg_c


def test_no_votes_triggers_fail() -> None:
    mod = _load_module()
    passed, no_votes, single_lf_only = mod.evaluate_snorkel_gate(
        {"success": 20, "single_lf_only": 12, "no_votes": 1}
    )
    assert passed is False
    assert no_votes == 1
    assert single_lf_only == 12


def test_single_lf_only_does_not_trigger_fail() -> None:
    mod = _load_module()
    passed, no_votes, single_lf_only = mod.evaluate_snorkel_gate(
        {"success": 20, "single_lf_only": 12, "no_votes": 0}
    )
    assert passed is True
    assert no_votes == 0
    assert single_lf_only == 12


def test_all_pass_scenario_produces_pass_banner() -> None:
    mod = _load_module()
    status = mod.overall_status_from_criteria(
        [
            {"status": "PASS"},
            {"status": "PASS"},
            {"status": "PASS"},
        ]
    )
    assert status == "PASS"
