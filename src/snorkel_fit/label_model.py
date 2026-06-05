from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from src.labeling_functions.base import LFInput, LabelingFunction, Vote

logger = logging.getLogger(__name__)


@dataclass
class _MajorityVoteModel:
    cardinality: int = 2

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        n_rows = int(L.shape[0])
        if n_rows == 0:
            return np.empty((0, self.cardinality), dtype=np.float64)

        probs = np.full((n_rows, self.cardinality), 1.0 / self.cardinality, dtype=np.float64)
        for row_index in range(n_rows):
            row = L[row_index]
            positives = int(np.sum(row == Vote.POSITIVE))
            negatives = int(np.sum(row == Vote.NEGATIVE))
            if positives == 0 and negatives == 0:
                probs[row_index] = np.array([0.5, 0.5], dtype=np.float64)
            elif positives > negatives:
                probs[row_index] = np.array([0.0, 1.0], dtype=np.float64)
            elif negatives > positives:
                probs[row_index] = np.array([1.0, 0.0], dtype=np.float64)
            else:
                probs[row_index] = np.array([0.5, 0.5], dtype=np.float64)
        return probs


def build_lf_vote_matrix(
    lfs: list[LabelingFunction],
    inputs: list[LFInput],
) -> tuple[np.ndarray, list[str]]:
    n_inputs = len(inputs)
    n_lfs = len(lfs)
    matrix = np.full((n_inputs, n_lfs), Vote.ABSTAIN, dtype=np.int8)
    lf_names = [str(lf.name) for lf in lfs]

    for lf_index, lf in enumerate(lfs):
        for input_index, lf_input in enumerate(inputs):
            output = lf(lf_input)
            matrix[input_index, lf_index] = int(output.vote)

    return matrix, lf_names


def fit_label_model(
    L: np.ndarray,
    cardinality: int = 2,
    seed: int = 42,
) -> Any:
    try:
        from snorkel.labeling.model import LabelModel  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        logger.warning("snorkel is unavailable; falling back to majority-vote model.")
        return _MajorityVoteModel(cardinality=cardinality)

    if int(L.shape[1]) < 3:
        logger.info(
            "Snorkel LabelModel requires >=3 LFs; got %s. Falling back to majority-vote model.",
            int(L.shape[1]),
        )
        return _MajorityVoteModel(cardinality=cardinality)

    lm = LabelModel(cardinality=cardinality, verbose=False)
    lm.fit(L, n_epochs=500, lr=0.01, seed=seed)
    return lm


def predict_probs(label_model: Any, L: np.ndarray) -> np.ndarray:
    probabilities = label_model.predict_proba(L)
    return np.asarray(probabilities, dtype=np.float64)


def _uniform_probs(n_rows: int) -> np.ndarray:
    if n_rows == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.full((n_rows, 2), 0.5, dtype=np.float64)


def _single_lf_probs(votes: np.ndarray) -> np.ndarray:
    n_rows = int(votes.shape[0])
    probabilities = _uniform_probs(n_rows)
    for row_index in range(n_rows):
        vote = int(votes[row_index])
        if vote == Vote.POSITIVE:
            probabilities[row_index] = np.array([0.0, 1.0], dtype=np.float64)
        elif vote == Vote.NEGATIVE:
            probabilities[row_index] = np.array([1.0, 0.0], dtype=np.float64)
        else:
            probabilities[row_index] = np.array([0.5, 0.5], dtype=np.float64)
    return probabilities


def _lf_polarity_stats(L: np.ndarray, lf_names: list[str]) -> dict[str, dict[str, float]]:
    n_rows = max(int(L.shape[0]), 1)
    output: dict[str, dict[str, float]] = {}
    for lf_index, lf_name in enumerate(lf_names):
        column = L[:, lf_index]
        n_pos = int(np.sum(column == Vote.POSITIVE))
        n_neg = int(np.sum(column == Vote.NEGATIVE))
        n_abs = int(np.sum(column == Vote.ABSTAIN))
        output[lf_name] = {
            "pct_positive": (n_pos / n_rows) * 100.0,
            "pct_negative": (n_neg / n_rows) * 100.0,
            "pct_abstain": (n_abs / n_rows) * 100.0,
        }
    return output


def _lf_coverage_stats(L: np.ndarray, lf_names: list[str]) -> dict[str, float]:
    n_rows = max(int(L.shape[0]), 1)
    output: dict[str, float] = {}
    for lf_index, lf_name in enumerate(lf_names):
        column = L[:, lf_index]
        n_firing = int(np.sum(column != Vote.ABSTAIN))
        output[lf_name] = (n_firing / n_rows) * 100.0
    return output


def _lf_overlap_stats(L: np.ndarray, lf_names: list[str]) -> dict[str, float | None]:
    overlaps: dict[str, float | None] = {}
    for left_idx, right_idx in combinations(range(len(lf_names)), 2):
        left_name = lf_names[left_idx]
        right_name = lf_names[right_idx]
        key = f"{left_name}__{right_name}"

        left_votes = L[:, left_idx]
        right_votes = L[:, right_idx]
        both_fire_mask = (left_votes != Vote.ABSTAIN) & (right_votes != Vote.ABSTAIN)
        n_overlap = int(np.sum(both_fire_mask))
        if n_overlap == 0:
            overlaps[key] = None
            continue
        agreement = np.sum(left_votes[both_fire_mask] == right_votes[both_fire_mask])
        overlaps[key] = (float(agreement) / float(n_overlap)) * 100.0
    return overlaps


def aggregate_predictions(
    lfs: list[LabelingFunction],
    inputs: list[LFInput],
    target_field: str,
    target_value: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    matched_lfs = [
        lf
        for lf in lfs
        if str(getattr(lf, "target_field", "")) == target_field
        and str(getattr(lf, "target_value", "")) == target_value
    ]

    n_rows = len(inputs)
    if not matched_lfs:
        diagnostics = {
            "lf_names_used": [],
            "lf_polarities": {},
            "lf_coverage": {},
            "lf_overlaps": {},
            "fit_status": "no_votes",
        }
        return _uniform_probs(n_rows), diagnostics

    L, lf_names = build_lf_vote_matrix(matched_lfs, inputs)
    lf_polarities = _lf_polarity_stats(L, lf_names)
    lf_coverage = _lf_coverage_stats(L, lf_names)
    lf_overlaps = _lf_overlap_stats(L, lf_names)

    has_any_votes = bool(np.any(L != Vote.ABSTAIN))
    if not has_any_votes:
        diagnostics = {
            "lf_names_used": lf_names,
            "lf_polarities": lf_polarities,
            "lf_coverage": lf_coverage,
            "lf_overlaps": lf_overlaps,
            "fit_status": "no_votes",
        }
        return _uniform_probs(n_rows), diagnostics

    covering_indices = [idx for idx in range(L.shape[1]) if np.any(L[:, idx] != Vote.ABSTAIN)]
    if len(covering_indices) <= 1:
        index = covering_indices[0]
        probabilities = _single_lf_probs(L[:, index])
        diagnostics = {
            "lf_names_used": lf_names,
            "lf_polarities": lf_polarities,
            "lf_coverage": lf_coverage,
            "lf_overlaps": lf_overlaps,
            "fit_status": "single_lf_only",
        }
        return probabilities, diagnostics

    model = fit_label_model(L=L, cardinality=2, seed=42)
    probabilities = predict_probs(model, L)
    diagnostics = {
        "lf_names_used": lf_names,
        "lf_polarities": lf_polarities,
        "lf_coverage": lf_coverage,
        "lf_overlaps": lf_overlaps,
        "fit_status": "success",
    }
    return probabilities, diagnostics
