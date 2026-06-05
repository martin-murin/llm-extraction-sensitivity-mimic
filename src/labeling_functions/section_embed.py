from __future__ import annotations

import logging

import numpy as np

from src.labeling_functions.embedding_backend import EmbeddingBackend, EmbeddingCache
from src.labeling_functions.section_parser import get_section, parse_sections
from src.schema.section_map import FIELD_SECTION_MAP

logger = logging.getLogger(__name__)


async def embed_notes_sections(
    notes: dict[int, str],
    fields: list[str],
    backend: EmbeddingBackend,
    cache: EmbeddingCache,
) -> dict[int, dict[str, np.ndarray]]:
    needed_sections: set[str] = set()
    for field in fields:
        if field not in FIELD_SECTION_MAP:
            raise KeyError(f"Unknown field in section embedding request: {field}")
        needed_sections.update(FIELD_SECTION_MAP[field])

    sorted_sections = sorted(needed_sections)
    section_requests: list[tuple[int, str, str]] = []
    result: dict[int, dict[str, np.ndarray]] = {int(hadm_id): {} for hadm_id in notes}

    for hadm_id, note_text in notes.items():
        sections = parse_sections(note_text)
        for section_name in sorted_sections:
            section_text = get_section(sections, section_name)
            if section_text is None:
                continue
            normalized = section_text.strip()
            if len(normalized) < 20:
                continue
            section_requests.append((int(hadm_id), section_name, normalized))

    if not section_requests:
        logger.info(
            "No sections met embedding criteria for fields=%s (backend=%s)",
            fields,
            backend.model_id,
        )
        return result

    section_texts = [entry[2] for entry in section_requests]
    embeddings = await cache.embed_cached(section_texts)

    for index, (hadm_id, section_name, _text) in enumerate(section_requests):
        result.setdefault(hadm_id, {})[section_name] = embeddings[index].astype(np.float32)

    logger.info(
        "Embedded %s note sections across %s notes using backend %s",
        len(section_requests),
        len(notes),
        backend.model_id,
    )
    return result
