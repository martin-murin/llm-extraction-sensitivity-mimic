from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.optimization.optimizer import (
    IterationLog,
    apply_revision,
    compute_cluster_kappa,
    run_guards,
    run_iteration_sequence,
    select_next_cluster,
)
from src.optimization.pattern_clustering import PatternCluster


def _cluster(cluster_id: str, total: int, affected_variant: str | None = "c") -> PatternCluster:
    return PatternCluster(
        cluster_id=cluster_id,
        cluster_label=cluster_id,
        affected_variant=affected_variant,
        member_fields=[
            {
                "field": "shock_present",
                "target_value": None,
                "kappa_mean": 0.1,
                "disagreement_count": total,
                "n_positive_total": 20,
            }
        ],
        total_disagreement_count=total,
        representative_examples=[],
    )


def _load_variant_a() -> str:
    return (config.REPO_ROOT / "src/schema/prompts/variant_a.md").read_text(encoding="utf-8")


def test_select_next_cluster_none_when_no_targetable_clusters() -> None:
    clusters = [_cluster("x", 100, affected_variant=None)]
    selected = select_next_cluster(clusters, history=[])
    assert selected is None


def test_select_next_cluster_returns_largest_targetable_cluster() -> None:
    clusters = [_cluster("small", 61), _cluster("large", 95), _cluster("medium", 80)]
    selected = select_next_cluster(clusters, history=[])
    assert selected is not None
    assert selected.cluster_id == "large"


def test_select_next_cluster_allows_previously_successful_cluster_if_still_largest() -> None:
    clusters = [_cluster("first", 120), _cluster("second", 110)]
    history = [
        IterationLog(
            iteration=1,
            cluster_id="first",
            targeted_variant="c",
            applied=True,
        )
    ]
    selected = select_next_cluster(clusters, history=history)
    assert selected is not None
    assert selected.cluster_id == "first"


def test_select_next_cluster_skips_at_or_below_min_disagreements() -> None:
    clusters = [_cluster("low", 50), _cluster("also_low", 49), _cluster("eligible", 51)]
    selected = select_next_cluster(clusters, history=[], min_disagreements=50)
    assert selected is not None
    assert selected.cluster_id == "eligible"


def test_select_next_cluster_none_when_none_above_threshold_regardless_of_history() -> None:
    clusters = [_cluster("low", 50), _cluster("also_low", 49)]
    history = [
        IterationLog(
            iteration=1,
            cluster_id="some_cluster",
            targeted_variant="c",
            applied=True,
        )
    ]
    selected = select_next_cluster(clusters, history=history, min_disagreements=50)
    assert selected is None


def test_compute_cluster_kappa_averages_member_fields() -> None:
    cluster = PatternCluster(
        cluster_id="c_cluster",
        cluster_label="c cluster",
        affected_variant="c",
        member_fields=[
            {"field": "shock_present", "target_value": None},
            {"field": "aki_present", "target_value": None},
        ],
        total_disagreement_count=120,
        representative_examples=[],
    )
    kappa_results = {
        "shock_present": {"kappa_mean": 0.4},
        "aki_present": {"kappa_mean": 0.8},
    }
    assert compute_cluster_kappa(cluster, kappa_results) == pytest.approx(0.6)


def test_compute_cluster_kappa_handles_target_value_keys() -> None:
    cluster = PatternCluster(
        cluster_id="c_tag_cluster",
        cluster_label="c tag cluster",
        affected_variant="c",
        member_fields=[
            {"field": "admission_reason_tags", "target_value": "cardiac_hf"},
        ],
        total_disagreement_count=70,
        representative_examples=[],
    )
    kappa_results = {
        "admission_reason_tags::cardiac_hf": {"kappa_mean": 0.77},
    }
    assert compute_cluster_kappa(cluster, kappa_results) == 0.77


def test_compute_cluster_kappa_returns_zero_when_missing_fields() -> None:
    cluster = PatternCluster(
        cluster_id="missing",
        cluster_label="missing",
        affected_variant="c",
        member_fields=[{"field": "does_not_exist", "target_value": None}],
        total_disagreement_count=90,
        representative_examples=[],
    )
    assert compute_cluster_kappa(cluster, {}) == 0.0


def test_run_guards_fails_when_vocabulary_sync_missing() -> None:
    original = _load_variant_a()
    revised = original.replace("cardiac_hf", "cardiachfremoved")
    passed, result = run_guards(revised, original)
    assert passed is False
    assert result["vocabulary_sync"]["passed"] is False


def test_run_guards_fails_when_reasoning_placeholder_missing() -> None:
    original = _load_variant_a()
    revised = original.replace("{{REASONING_INSTRUCTIONS}}", "", 1)
    passed, result = run_guards(revised, original)
    assert passed is False
    assert result["reasoning_placeholder_present"]["passed"] is False


def test_run_guards_fails_when_edit_distance_too_high() -> None:
    original = _load_variant_a()
    revised = "x" * len(original)
    passed, result = run_guards(revised, original)
    assert passed is False
    assert result["edit_distance"]["passed"] is False


def test_run_guards_fails_when_frozen_anchor_missing() -> None:
    original = _load_variant_a()
    revised = original.replace("Edge cases", "Case notes", 1)
    passed, result = run_guards(revised, original)
    assert passed is False
    assert result["frozen_content"]["passed"] is False


def test_run_guards_passes_for_original_prompt() -> None:
    original = _load_variant_a()
    passed, result = run_guards(original, original)
    assert passed is True
    assert result["all_passed"] is True


def test_apply_revision_writes_prompt_file(tmp_path: Path) -> None:
    revised_text = "SYNTHETIC_PROMPT"
    written = apply_revision("a", revised_text, tmp_path)
    assert written == tmp_path / "variant_a.md"
    assert written.read_text(encoding="utf-8") == revised_text


def test_loop_integration_success_then_failure_then_continue() -> None:
    original = _load_variant_a()
    clusters = [_cluster("cluster_one", 100), _cluster("cluster_two", 90)]
    applied_clusters: list[str] = []

    def revision_provider(cluster: PatternCluster, iteration: int) -> str:
        if iteration == 2:
            return original.replace("{{REASONING_INSTRUCTIONS}}", "", 1)
        return original

    def apply_callback(cluster: PatternCluster, revised_prompt: str) -> None:
        applied_clusters.append(cluster.cluster_id)
        assert revised_prompt == original

    history = run_iteration_sequence(
        clusters=clusters,
        original_prompt=original,
        revision_provider=revision_provider,
        apply_callback=apply_callback,
        max_iterations=3,
    )

    assert len(history) == 3
    assert history[0].applied is True
    assert history[1].applied is False
    assert history[2].applied is True
    assert applied_clusters == ["cluster_one", "cluster_one"]


def test_loop_marks_addressed_but_ineffective_and_continues() -> None:
    original = _load_variant_a()
    clusters = [_cluster("cluster_one", 120), _cluster("cluster_two", 110)]
    applied_clusters: list[str] = []

    def revision_provider(cluster: PatternCluster, iteration: int) -> str:
        return original

    def apply_callback(cluster: PatternCluster, revised_prompt: str) -> None:
        applied_clusters.append(cluster.cluster_id)
        assert revised_prompt == original

    history = run_iteration_sequence(
        clusters=clusters,
        original_prompt=original,
        revision_provider=revision_provider,
        apply_callback=apply_callback,
        max_iterations=3,
        cluster_delta_provider=lambda _cluster, iteration: 1.0 if iteration == 1 else 3.0,
        kappa_plateau_pp=2.0,
    )

    assert len(history) == 3
    assert history[0].applied is True
    assert history[0].effectiveness == "addressed_but_ineffective"
    assert history[1].applied is True
    assert history[1].effectiveness == "successful"
    assert history[2].applied is True
    assert history[2].effectiveness == "successful"
    assert applied_clusters == ["cluster_one", "cluster_one", "cluster_one"]


def test_loop_stops_when_no_cluster_above_threshold_remains() -> None:
    original = _load_variant_a()
    clusters = [_cluster("too_small", 49), _cluster("at_threshold", 50)]
    applied_clusters: list[str] = []

    def revision_provider(cluster: PatternCluster, iteration: int) -> str:
        return original

    def apply_callback(cluster: PatternCluster, revised_prompt: str) -> None:
        applied_clusters.append(cluster.cluster_id)
        assert revised_prompt == original

    history = run_iteration_sequence(
        clusters=clusters,
        original_prompt=original,
        revision_provider=revision_provider,
        apply_callback=apply_callback,
        max_iterations=3,
        min_disagreements=50,
    )

    assert history == []
    assert applied_clusters == []


def test_loop_stops_after_two_consecutive_failed_iterations() -> None:
    original = _load_variant_a()
    clusters = [_cluster("cluster_one", 120), _cluster("cluster_two", 110)]

    def revision_provider(cluster: PatternCluster, iteration: int) -> str:
        return original.replace("{{REASONING_INSTRUCTIONS}}", "", 1)

    history = run_iteration_sequence(
        clusters=clusters,
        original_prompt=original,
        revision_provider=revision_provider,
        apply_callback=lambda _cluster, _prompt: None,
        max_iterations=5,
    )

    assert len(history) == 2
    assert history[0].applied is False
    assert history[1].applied is False
