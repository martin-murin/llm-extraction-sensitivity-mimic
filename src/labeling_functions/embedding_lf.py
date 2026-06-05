from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.labeling_functions.base import LFInput, LFOutput, LabelingFunction, Vote
from src.labeling_functions.embedding_backend import EmbeddingBackend, EmbeddingCache
from src.labeling_functions.regex_lf import load_pattern_yaml
from src.schema.section_map import FIELD_SECTION_MAP

logger = logging.getLogger(__name__)

# Embedding LFs are parked as of 2026-04-24. See docs/reports/embedding_lf_findings.md.
# The loader below returns an empty list by default. To re-enable, set
# EMBEDDING_LFS_ENABLED = True. The underlying code remains functional for future
# experiments with clinical-domain embedding backends.
EMBEDDING_LFS_ENABLED: bool = False
_DISABLED_LOGGED: bool = False


@dataclass
class EmbeddingLabelingFunction:
    name: str
    target_field: str
    target_value: str | None
    seed_phrases: list[str]
    seed_phrase_embeddings: np.ndarray
    threshold: float

    def __call__(self, inputs: LFInput) -> LFOutput:
        if inputs.section_embeddings is None:
            return LFOutput(vote=Vote.ABSTAIN, evidence="no_section_or_short")

        required_sections = FIELD_SECTION_MAP[self.target_field]
        section_vectors: list[tuple[str, np.ndarray]] = []
        for section_name in required_sections:
            vector = inputs.section_embeddings.get(section_name)
            if not isinstance(vector, np.ndarray):
                continue
            section_vectors.append((section_name, vector.astype(np.float32)))

        if not section_vectors:
            return LFOutput(vote=Vote.ABSTAIN, evidence="no_section_or_short")

        best_similarity = float("-inf")
        best_seed_index = -1
        best_section = ""

        for section_name, section_vector in section_vectors:
            if section_vector.ndim != 1:
                section_vector = section_vector.reshape(-1)
            section_norm = float(np.linalg.norm(section_vector))
            if section_norm == 0.0:
                continue
            normalized_section = (section_vector / section_norm).astype(np.float32)

            similarities = self.seed_phrase_embeddings @ normalized_section
            if similarities.size == 0:
                continue
            local_index = int(np.argmax(similarities))
            local_similarity = float(similarities[local_index])
            if local_similarity > best_similarity:
                best_similarity = local_similarity
                best_seed_index = local_index
                best_section = section_name

        if best_seed_index < 0:
            return LFOutput(vote=Vote.ABSTAIN, evidence="no_section_or_short")

        if best_similarity >= self.threshold:
            return LFOutput(
                vote=Vote.POSITIVE,
                confidence=best_similarity,
                evidence=(
                    f"embed sim={best_similarity:.3f} >= {self.threshold:.3f}; "
                    f"seed_idx={best_seed_index}; section={best_section}"
                ),
            )

        return LFOutput(
            vote=Vote.ABSTAIN,
            confidence=best_similarity,
            evidence=(
                f"embed sim={best_similarity:.3f} < {self.threshold:.3f}; "
                f"seed_idx={best_seed_index}; section={best_section}"
            ),
        )


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def build_embedding_lf(
    field_name: str,
    target_value: str,
    seed_phrases: list[str],
    threshold: float,
    seed_phrase_embeddings: np.ndarray,
) -> EmbeddingLabelingFunction:
    if field_name not in FIELD_SECTION_MAP:
        raise KeyError(f"Unknown field for embedding LF: {field_name}")

    embeddings = np.asarray(seed_phrase_embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("seed_phrase_embeddings must be a 2D matrix (K, D)")
    if embeddings.shape[0] != len(seed_phrases):
        raise ValueError(
            "seed_phrase_embeddings row count must match number of seed phrases: "
            f"{embeddings.shape[0]} != {len(seed_phrases)}"
        )

    return EmbeddingLabelingFunction(
        name=f"embed_{field_name}_{target_value}",
        target_field=field_name,
        target_value=target_value,
        seed_phrases=list(seed_phrases),
        seed_phrase_embeddings=_l2_normalize_rows(embeddings),
        threshold=float(threshold),
    )


def build_all_embedding_lfs(
    patterns_dir: Path,
    backend: EmbeddingBackend,
    cache: EmbeddingCache,
) -> list[LabelingFunction]:
    global _DISABLED_LOGGED
    if not EMBEDDING_LFS_ENABLED:
        if not _DISABLED_LOGGED:
            logger.info(
                "Embedding LFs are currently disabled "
                "(see docs/reports/embedding_lf_findings.md)."
            )
            _DISABLED_LOGGED = True
        return []

    if not patterns_dir.exists():
        logger.info("Patterns directory does not exist: %s", patterns_dir)
        return []

    lfs: list[LabelingFunction] = []
    for yaml_path in sorted(patterns_dir.glob("*.yaml")):
        payload = load_pattern_yaml(yaml_path)

        field_name = str(payload["field_name"])
        target_value = str(payload["target_value"])
        seed_phrases_raw = payload.get("embedding_seed_phrases", [])
        if not isinstance(seed_phrases_raw, list):
            raise ValueError(f"embedding_seed_phrases must be a list in {yaml_path}")

        seed_phrases = [str(item) for item in seed_phrases_raw if str(item).strip()]
        if not seed_phrases:
            logger.info(
                "Skipping embedding LF for %s (%s): no seed phrases",
                field_name,
                yaml_path.name,
            )
            continue

        threshold_raw = payload.get("embedding_threshold")
        if threshold_raw is None:
            logger.info(
                "Skipping embedding LF for %s (%s): missing embedding_threshold",
                field_name,
                yaml_path.name,
            )
            continue
        if not isinstance(threshold_raw, (int, float)):
            raise ValueError(f"embedding_threshold must be numeric in {yaml_path}")

        seed_phrase_embeddings = asyncio.run(cache.embed_cached(seed_phrases))
        lf = build_embedding_lf(
            field_name=field_name,
            target_value=target_value,
            seed_phrases=seed_phrases,
            threshold=float(threshold_raw),
            seed_phrase_embeddings=seed_phrase_embeddings,
        )
        lfs.append(lf)

    logger.info(
        "Loaded %s embedding LFs from %s using backend %s",
        len(lfs),
        patterns_dir,
        backend.model_id,
    )
    return lfs
