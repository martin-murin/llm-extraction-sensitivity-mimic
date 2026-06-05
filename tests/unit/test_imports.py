from __future__ import annotations


def test_imports_and_model_id() -> None:
    from src import config
    from src.db.connection import discover_schemas, get_engine
    from src.db.icd_utils import icd10_chapter_from_code
    from src.db.queries import (
        count_notes,
        fetch_icd_codes_by_hadm_ids,
        fetch_notes_by_hadm_ids,
        fetch_primary_icd_by_hadm_ids,
        join_cardinality,
        note_length_stats,
        pull_split_candidates,
        sample_redaction_excerpts,
        top_any_position_icds,
        top_primary_icds,
    )
    from src.io.splits import build_stratified_splits, save_splits
    from src.labeling_functions.base import LFRegistry, LabelingFunction, LFInput, LFOutput, Vote
    from src.labeling_functions.embedding_backend import (
        EmbeddingBackend,
        EmbeddingCache,
        OpenAIEmbeddingBackend,
    )
    from src.labeling_functions.embedding_lf import (
        EmbeddingLabelingFunction,
        build_all_embedding_lfs,
        build_embedding_lf,
    )
    from src.labeling_functions.icd_lf import ICD_LF_SPECS, build_all_icd_lfs, build_icd_lf
    from src.labeling_functions.llm_lf import (
        ACTIVE_TRISTATE_FIELDS_FOR_SNORKEL,
        ICD_ANCHORED_ADMISSION_TAGS,
        SNORKEL_TARGET_FIELD_VALUE_PAIRS,
        FieldType,
        LLMLabelingFunction,
        build_all_llm_lfs,
        build_llm_lf,
    )
    from src.labeling_functions.pattern_bootstrap import (
        REGEX_PILOT_FIELDS,
        derive_embedding_seed_phrases,
        derive_regex_patterns,
        extract_anchor_phrases,
        load_coverage_v2_results,
        write_pattern_yaml,
    )
    from src.labeling_functions.regex_lf import (
        NEGATION_CUES,
        NEGATION_WINDOW_CHARS,
        build_all_regex_lfs,
        build_regex_lf,
        is_negated,
        load_pattern_yaml,
    )
    from src.labeling_functions.section_parser import (
        SECTION_ALIASES,
        coverage_report,
        get_section,
        parse_sections,
    )
    from src.labeling_functions.section_embed import embed_notes_sections
    from src.llm.batch_runner import BatchSummary, run_batch
    from src.llm.client import LLMClient
    from src.llm.extractor import (
        ExtractionResult,
        PromptSchemaDriftError,
        build_response_format,
        build_strict_json_schema,
        build_messages,
        check_prompt_vocabulary_sync,
        count_prompt_tokens,
        extract_content_from_raw_response,
        extract_note,
        load_prompt_template,
    )
    from src.schema.section_map import FIELD_SECTION_MAP
    from src.snorkel_fit.label_model import (
        aggregate_predictions,
        build_lf_vote_matrix,
        fit_label_model,
        predict_probs,
    )
    from src.schema.vocabulary import CHAPTER_TO_PLAUSIBLE_TAGS
    from src.utils.logging import BudgetExceededError, CostTracker, get_logger

    assert callable(get_engine)
    assert callable(discover_schemas)
    assert callable(count_notes)
    assert callable(note_length_stats)
    assert callable(top_primary_icds)
    assert callable(top_any_position_icds)
    assert callable(join_cardinality)
    assert callable(sample_redaction_excerpts)
    assert callable(pull_split_candidates)
    assert callable(fetch_notes_by_hadm_ids)
    assert callable(fetch_icd_codes_by_hadm_ids)
    assert callable(fetch_primary_icd_by_hadm_ids)
    assert callable(icd10_chapter_from_code)
    assert callable(build_stratified_splits)
    assert callable(save_splits)
    assert Vote is not None
    assert LFInput is not None
    assert LFOutput is not None
    assert LabelingFunction is not None
    assert LFRegistry is not None
    assert EmbeddingBackend is not None
    assert EmbeddingCache is not None
    assert OpenAIEmbeddingBackend is not None
    assert EmbeddingLabelingFunction is not None
    assert callable(build_embedding_lf)
    assert callable(build_all_embedding_lfs)
    assert callable(build_icd_lf)
    assert callable(build_all_icd_lfs)
    assert isinstance(ICD_LF_SPECS, list)
    assert FieldType is not None
    assert LLMLabelingFunction is not None
    assert callable(build_llm_lf)
    assert callable(build_all_llm_lfs)
    assert isinstance(ICD_ANCHORED_ADMISSION_TAGS, tuple)
    assert isinstance(ACTIVE_TRISTATE_FIELDS_FOR_SNORKEL, tuple)
    assert isinstance(SNORKEL_TARGET_FIELD_VALUE_PAIRS, list)
    assert callable(load_coverage_v2_results)
    assert callable(extract_anchor_phrases)
    assert callable(derive_regex_patterns)
    assert callable(derive_embedding_seed_phrases)
    assert callable(write_pattern_yaml)
    assert isinstance(REGEX_PILOT_FIELDS, list)
    assert callable(load_pattern_yaml)
    assert callable(is_negated)
    assert callable(build_regex_lf)
    assert callable(build_all_regex_lfs)
    assert isinstance(NEGATION_CUES, tuple)
    assert NEGATION_WINDOW_CHARS > 0
    assert callable(parse_sections)
    assert callable(get_section)
    assert callable(coverage_report)
    assert isinstance(SECTION_ALIASES, dict)
    assert callable(embed_notes_sections)
    assert callable(run_batch)
    assert BatchSummary is not None
    assert callable(load_prompt_template)
    assert callable(check_prompt_vocabulary_sync)
    assert callable(build_messages)
    assert callable(count_prompt_tokens)
    assert callable(build_strict_json_schema)
    assert callable(build_response_format)
    assert callable(extract_content_from_raw_response)
    assert callable(extract_note)
    assert ExtractionResult is not None
    assert PromptSchemaDriftError is not None
    assert isinstance(FIELD_SECTION_MAP, dict)
    assert callable(build_lf_vote_matrix)
    assert callable(fit_label_model)
    assert callable(predict_probs)
    assert callable(aggregate_predictions)
    assert isinstance(CHAPTER_TO_PLAUSIBLE_TAGS, dict)
    assert LLMClient is not None
    assert callable(get_logger)
    assert CostTracker is not None
    assert BudgetExceededError is not None
    assert config.MODEL_ID == "gpt-5.4-nano-2026-03-17"
