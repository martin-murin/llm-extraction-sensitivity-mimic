from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.labeling_functions.base import LFInput, LFOutput, LabelingFunction, Vote
from src.snorkel_fit.label_model import (
    aggregate_predictions,
    build_lf_vote_matrix,
    fit_label_model,
    predict_probs,
)


@dataclass
class _DummyLF:
    name: str
    target_field: str
    target_value: str | None
    votes_by_hadm: dict[int, Vote]

    def __call__(self, inputs: LFInput) -> LFOutput:
        return LFOutput(vote=self.votes_by_hadm.get(inputs.hadm_id, Vote.ABSTAIN))


def _inputs() -> list[LFInput]:
    return [
        LFInput(hadm_id=1, note_text="n1"),
        LFInput(hadm_id=2, note_text="n2"),
        LFInput(hadm_id=3, note_text="n3"),
    ]


def test_build_lf_vote_matrix_shape() -> None:
    lfs: list[LabelingFunction] = [
        _DummyLF("lf1", "aki_present", "yes", {1: Vote.POSITIVE, 2: Vote.NEGATIVE}),
        _DummyLF("lf2", "aki_present", "yes", {2: Vote.POSITIVE}),
    ]
    L, names = build_lf_vote_matrix(lfs, _inputs())
    assert L.shape == (3, 2)
    assert names == ["lf1", "lf2"]


def test_build_lf_vote_matrix_vote_translation() -> None:
    lfs: list[LabelingFunction] = [
        _DummyLF("lf1", "aki_present", "yes", {1: Vote.POSITIVE, 2: Vote.NEGATIVE, 3: Vote.ABSTAIN})
    ]
    L, _ = build_lf_vote_matrix(lfs, _inputs())
    assert L.tolist() == [[1], [0], [-1]]


def test_fit_label_model_predicts() -> None:
    L = np.array([[1, 1], [0, 0], [1, -1], [0, -1]], dtype=np.int8)
    model = fit_label_model(L, cardinality=2, seed=42)
    probs = predict_probs(model, L)
    assert probs.shape == (4, 2)


def test_aggregate_predictions_probabilities_sum_to_one() -> None:
    lfs: list[LabelingFunction] = [
        _DummyLF("lf1", "aki_present", "yes", {1: Vote.POSITIVE, 2: Vote.NEGATIVE}),
        _DummyLF("lf2", "aki_present", "yes", {1: Vote.POSITIVE, 3: Vote.POSITIVE}),
    ]
    probs, diagnostics = aggregate_predictions(lfs, _inputs(), "aki_present", "yes")
    assert probs.shape == (3, 2)
    assert np.allclose(probs.sum(axis=1), np.array([1.0, 1.0, 1.0]))
    assert diagnostics["fit_status"] in {"success", "single_lf_only"}


def test_aggregate_predictions_handles_all_abstain() -> None:
    lfs: list[LabelingFunction] = [
        _DummyLF("lf1", "aki_present", "yes", {}),
        _DummyLF("lf2", "aki_present", "yes", {}),
    ]
    probs, diagnostics = aggregate_predictions(lfs, _inputs(), "aki_present", "yes")
    assert diagnostics["fit_status"] == "no_votes"
    assert np.allclose(probs, np.full((3, 2), 0.5))


def test_aggregate_predictions_handles_single_covering_lf() -> None:
    lfs: list[LabelingFunction] = [
        _DummyLF("lf1", "aki_present", "yes", {1: Vote.POSITIVE, 2: Vote.NEGATIVE}),
        _DummyLF("lf2", "aki_present", "yes", {}),
    ]
    probs, diagnostics = aggregate_predictions(lfs, _inputs(), "aki_present", "yes")
    assert diagnostics["fit_status"] == "single_lf_only"
    assert np.allclose(probs[0], np.array([0.0, 1.0]))
    assert np.allclose(probs[1], np.array([1.0, 0.0]))
    assert np.allclose(probs[2], np.array([0.5, 0.5]))
