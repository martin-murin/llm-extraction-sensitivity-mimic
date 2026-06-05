from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from src.schema.fields import LLMNoteFeatures


class Vote(IntEnum):
    POSITIVE = 1
    NEGATIVE = 0
    ABSTAIN = -1


@dataclass(frozen=True)
class LFInput:
    hadm_id: int
    note_text: str
    icd_codes: list[tuple[str, int]] | None = None
    primary_icd_code: str | None = None
    primary_icd_version: int | None = None
    sections: dict[str, str] | None = None
    section_embeddings: dict[str, np.ndarray] | None = None
    llm_extraction_by_variant: dict[str, LLMNoteFeatures] | None = None


@dataclass(frozen=True)
class LFOutput:
    vote: Vote
    confidence: float | None = None
    evidence: str | None = None


class LabelingFunction(Protocol):
    """Protocol all labeling functions implement."""

    name: str
    target_field: str
    target_value: str | None = None

    def __call__(self, inputs: LFInput) -> LFOutput: ...


class LFRegistry:
    def __init__(self) -> None:
        self._lfs: list[LabelingFunction] = []

    def register(self, lf: LabelingFunction) -> None:
        if any(existing.name == lf.name for existing in self._lfs):
            raise ValueError(f"Labeling function with name '{lf.name}' already registered.")
        self._lfs.append(lf)

    def for_field(self, field_name: str) -> list[LabelingFunction]:
        return [lf for lf in self._lfs if lf.target_field == field_name]

    def all(self) -> list[LabelingFunction]:
        return list(self._lfs)

    def __len__(self) -> int:
        return len(self._lfs)
