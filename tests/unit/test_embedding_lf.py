from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from src.labeling_functions.base import LFInput, Vote
from src.labeling_functions.embedding_backend import EmbeddingBackend, EmbeddingCache
from src.labeling_functions.embedding_lf import build_all_embedding_lfs, build_embedding_lf


class _FakeBackend(EmbeddingBackend):
    def __init__(self, model_id: str = "fake-model", dimension: int = 3) -> None:
        self._model_id = model_id
        self._dimension = dimension
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        rows: list[np.ndarray] = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                rows.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
            elif "beta" in lowered:
                rows.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
            else:
                rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        matrix = np.vstack(rows).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (matrix / norms).astype(np.float32)


def _run(coro: object) -> np.ndarray:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_embedding_lf_abstains_when_section_embeddings_missing() -> None:
    lf = build_embedding_lf(
        field_name="home_health_ordered",
        target_value="yes",
        seed_phrases=["alpha"],
        threshold=0.8,
        seed_phrase_embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    output = lf(LFInput(hadm_id=1, note_text="note", section_embeddings=None))
    assert output.vote == Vote.ABSTAIN


def test_embedding_lf_abstains_when_required_sections_missing() -> None:
    lf = build_embedding_lf(
        field_name="home_health_ordered",
        target_value="yes",
        seed_phrases=["alpha"],
        threshold=0.8,
        seed_phrase_embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    output = lf(
        LFInput(
            hadm_id=2,
            note_text="note",
            section_embeddings={
                "Brief Hospital Course": np.array([1.0, 0.0, 0.0], dtype=np.float32)
            },
        )
    )
    assert output.vote == Vote.ABSTAIN


def test_embedding_lf_votes_positive_when_similarity_crosses_threshold() -> None:
    lf = build_embedding_lf(
        field_name="home_health_ordered",
        target_value="yes",
        seed_phrases=["alpha"],
        threshold=0.8,
        seed_phrase_embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    output = lf(
        LFInput(
            hadm_id=3,
            note_text="note",
            section_embeddings={
                "Discharge Instructions": np.array([0.95, 0.1, 0.0], dtype=np.float32)
            },
        )
    )
    assert output.vote == Vote.POSITIVE


def test_embedding_lf_abstains_below_threshold() -> None:
    lf = build_embedding_lf(
        field_name="home_health_ordered",
        target_value="yes",
        seed_phrases=["alpha"],
        threshold=0.8,
        seed_phrase_embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    output = lf(
        LFInput(
            hadm_id=4,
            note_text="note",
            section_embeddings={
                "Discharge Instructions": np.array([0.2, 0.2, 0.0], dtype=np.float32)
            },
        )
    )
    assert output.vote == Vote.ABSTAIN


def test_embedding_lf_uses_max_similarity_across_sections() -> None:
    lf = build_embedding_lf(
        field_name="home_health_ordered",
        target_value="yes",
        seed_phrases=["alpha"],
        threshold=0.8,
        seed_phrase_embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    output = lf(
        LFInput(
            hadm_id=5,
            note_text="note",
            section_embeddings={
                "Discharge Instructions": np.array([0.1, 0.9, 0.0], dtype=np.float32),
                "Discharge Disposition": np.array([0.9, 0.1, 0.0], dtype=np.float32),
            },
        )
    )
    assert output.vote == Vote.POSITIVE


def test_build_all_embedding_lfs_skips_yaml_with_empty_seed_phrases(tmp_path: Path) -> None:
    payload_skip = {
        "field_name": "cardiac_rehab_referred",
        "target_value": "yes",
        "regex_patterns": [r"\\bcardiac rehab\\b"],
        "embedding_seed_phrases": [],
        "embedding_threshold": 0.6,
    }
    payload_keep = {
        "field_name": "home_health_ordered",
        "target_value": "yes",
        "regex_patterns": [r"\\bvisiting nurse\\b"],
        "embedding_seed_phrases": ["alpha phrase"],
        "embedding_threshold": 0.6,
    }

    (tmp_path / "cardiac_rehab_referred__yes.yaml").write_text(
        yaml.safe_dump(payload_skip, sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "home_health_ordered__yes.yaml").write_text(
        yaml.safe_dump(payload_keep, sort_keys=False),
        encoding="utf-8",
    )

    backend = _FakeBackend()
    cache = EmbeddingCache(cache_dir=tmp_path / "cache", backend=backend)
    with patch("src.labeling_functions.embedding_lf.EMBEDDING_LFS_ENABLED", True):
        lfs = build_all_embedding_lfs(tmp_path, backend=backend, cache=cache)
    assert len(lfs) == 1
    assert lfs[0].target_field == "home_health_ordered"


def test_build_all_embedding_lfs_disabled_by_default(tmp_path: Path) -> None:
    payload_keep = {
        "field_name": "home_health_ordered",
        "target_value": "yes",
        "regex_patterns": [r"\\bvisiting nurse\\b"],
        "embedding_seed_phrases": ["alpha phrase"],
        "embedding_threshold": 0.6,
    }
    (tmp_path / "home_health_ordered__yes.yaml").write_text(
        yaml.safe_dump(payload_keep, sort_keys=False),
        encoding="utf-8",
    )

    backend = _FakeBackend()
    cache = EmbeddingCache(cache_dir=tmp_path / "cache", backend=backend)
    lfs = build_all_embedding_lfs(tmp_path, backend=backend, cache=cache)
    assert lfs == []


def test_embedding_cache_miss_computes_and_stores(tmp_path: Path) -> None:
    backend = _FakeBackend()
    cache = EmbeddingCache(cache_dir=tmp_path / "cache", backend=backend)

    vectors = _run(cache.embed_cached(["alpha text"]))
    assert vectors.shape == (1, backend.dimension)
    assert backend.calls == 1
    assert cache.cache_misses == 1
    files = list((tmp_path / "cache").glob("*.npy"))
    assert len(files) == 1


def test_embedding_cache_hit_avoids_backend_call(tmp_path: Path) -> None:
    backend = _FakeBackend()
    cache = EmbeddingCache(cache_dir=tmp_path / "cache", backend=backend)

    _run(cache.embed_cached(["alpha text"]))
    assert backend.calls == 1

    _run(cache.embed_cached(["alpha text"]))
    assert backend.calls == 1
    assert cache.cache_hits == 1


def test_embedding_cache_key_differs_by_model_id(tmp_path: Path) -> None:
    backend_a = _FakeBackend(model_id="fake-a")
    backend_b = _FakeBackend(model_id="fake-b")
    cache_a = EmbeddingCache(cache_dir=tmp_path / "cache_a", backend=backend_a)
    cache_b = EmbeddingCache(cache_dir=tmp_path / "cache_b", backend=backend_b)

    assert cache_a._key("shared text") != cache_b._key("shared text")


def test_calibration_threshold_floor_is_065() -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "10_calibrate_embedding_thresholds.py"
    )
    spec = importlib.util.spec_from_file_location("calibrate_thresholds", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    threshold, reason = module.calibrated_threshold_from_p10(0.22)
    assert threshold == 0.65
    assert reason == "floor_0.65"
