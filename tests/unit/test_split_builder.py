from __future__ import annotations

import importlib.util

import pandas as pd  # type: ignore[import-untyped]

from src import config


def _load_script_module(script_name: str, module_name: str):
    script_path = config.REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_methodology_split_filter_skips_existing_500_hadm_ids() -> None:
    module = _load_script_module("19_build_1k_split.py", "build_1k_split")
    candidates = pd.DataFrame(
        {
            "hadm_id": [1001, 1002, 1003, 1004],
            "subject_id": [1, 2, 3, 4],
        }
    )
    excluded = {1002, 1004}

    filtered = module.filter_candidates_excluding_hadm_ids(candidates, excluded)
    filtered_ids = set(filtered["hadm_id"].astype(int).tolist())

    assert filtered_ids == {1001, 1003}
    assert filtered_ids.isdisjoint(excluded)
