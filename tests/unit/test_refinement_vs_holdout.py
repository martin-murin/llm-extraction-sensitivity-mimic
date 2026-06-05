from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "17_refinement_vs_holdout.py"
    spec = importlib.util.spec_from_file_location("script17_refinement_vs_holdout", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sidecar(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def test_load_sidecar_reads_valid_json(tmp_path: Path) -> None:
    module = _load_script_module()
    sidecar_path = tmp_path / "ref.json"
    payload = {
        "run_ids": ["a", "b", "c"],
        "kappa_summary_filtered": {"overall_median_kappa": 0.7},
        "kappa_results": {
            "shock_present": {
                "kappa_mean": 0.7,
                "kappa_A_B": 0.7,
                "kappa_A_C": 0.7,
                "kappa_B_C": 0.7,
                "low_base_rate_flag": False,
            }
        },
    }
    _write_sidecar(sidecar_path, payload)
    loaded = module._load_sidecar(sidecar_path)
    assert loaded["kappa_results"]["shock_present"]["kappa_mean"] == 0.7


def test_field_present_only_in_one_input_flagged_but_no_crash(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    refinement_path = tmp_path / "ref.json"
    holdout_path = tmp_path / "hold.json"
    output_path = tmp_path / "comparison.md"

    _write_sidecar(
        refinement_path,
        {
            "run_ids": ["ref_a", "ref_b", "ref_c"],
            "kappa_summary_filtered": {"overall_median_kappa": 0.71},
            "kappa_results": {
                "shock_present": {
                    "kappa_mean": 0.7,
                    "kappa_A_B": 0.7,
                    "kappa_A_C": 0.7,
                    "kappa_B_C": 0.7,
                    "low_base_rate_flag": False,
                },
                "only_ref": {
                    "kappa_mean": 0.6,
                    "kappa_A_B": 0.6,
                    "kappa_A_C": 0.6,
                    "kappa_B_C": 0.6,
                    "low_base_rate_flag": False,
                },
            },
        },
    )
    _write_sidecar(
        holdout_path,
        {
            "run_ids": ["hold_a", "hold_b", "hold_c"],
            "kappa_summary_filtered": {"overall_median_kappa": 0.69},
            "kappa_results": {
                "shock_present": {
                    "kappa_mean": 0.68,
                    "kappa_A_B": 0.68,
                    "kappa_A_C": 0.68,
                    "kappa_B_C": 0.68,
                    "low_base_rate_flag": False,
                },
                "only_hold": {
                    "kappa_mean": 0.5,
                    "kappa_A_B": 0.5,
                    "kappa_A_C": 0.5,
                    "kappa_B_C": 0.5,
                    "low_base_rate_flag": False,
                },
            },
        },
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "17_refinement_vs_holdout.py",
            "--refinement-kappa",
            str(refinement_path),
            "--holdout-kappa",
            str(holdout_path),
            "--output",
            str(output_path),
        ],
    )
    assert module.main() == 0
    text = output_path.read_text(encoding="utf-8")
    assert "refinement_only_count" in text
    assert "holdout_only_count" in text
    assert "only_ref" in text
    assert "only_hold" in text


def test_median_delta_handles_empty_shared_input_gracefully(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    refinement_path = tmp_path / "ref_empty.json"
    holdout_path = tmp_path / "hold_empty.json"
    output_path = tmp_path / "comparison_empty.md"

    _write_sidecar(
        refinement_path,
        {
            "run_ids": ["ref_a", "ref_b", "ref_c"],
            "kappa_summary_filtered": {"overall_median_kappa": 0.0},
            "kappa_results": {},
        },
    )
    _write_sidecar(
        holdout_path,
        {
            "run_ids": ["hold_a", "hold_b", "hold_c"],
            "kappa_summary_filtered": {"overall_median_kappa": 0.0},
            "kappa_results": {},
        },
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "17_refinement_vs_holdout.py",
            "--refinement-kappa",
            str(refinement_path),
            "--holdout-kappa",
            str(holdout_path),
            "--output",
            str(output_path),
        ],
    )
    assert module.main() == 0
    text = output_path.read_text(encoding="utf-8")
    assert "n_shared_fields" in text
    assert "| 0 | 0 |" in text or "_No rows._" in text
