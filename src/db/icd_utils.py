from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

UNMAPPED = "ZZ. Unmapped"


def icd10_chapter_from_code(code: str, version: int) -> str:
    raw_code = (code or "").strip().upper()
    if not raw_code:
        logger.debug("Unmapped ICD code: empty input", extra={"code": code, "version": version})
        return UNMAPPED

    try:
        version_int = int(version)
    except (TypeError, ValueError):
        logger.debug(
            "Unmapped ICD code: invalid version",
            extra={"code": code, "version": version},
        )
        return UNMAPPED

    if version_int == 10:
        return _icd10_chapter(raw_code)
    if version_int == 9:
        return _icd9_chapter(raw_code)

    logger.debug(
        "Unmapped ICD code: unsupported version",
        extra={"code": code, "version": version},
    )
    return UNMAPPED


def _icd10_chapter(code: str) -> str:
    compact = code.replace(".", "")
    match = re.match(r"^([A-Z])(\d{2})", compact)
    if not match:
        logger.debug("Unmapped ICD-10 code format", extra={"code": code})
        return UNMAPPED

    letter = match.group(1)
    category = int(match.group(2))

    if letter in {"A", "B"}:
        return "I. Infectious"
    if letter == "C":
        return "II. Neoplasms"
    if letter == "D":
        if category <= 48:
            return "II. Neoplasms"
        if 50 <= category <= 89:
            return "III. Blood/immune"
        logger.debug("Unmapped ICD-10 D-code category", extra={"code": code})
        return UNMAPPED
    if letter == "E":
        return "IV. Endocrine/metabolic"
    if letter == "F":
        return "V. Mental"
    if letter == "G":
        return "VI. Nervous"
    if letter == "H":
        if category <= 59:
            return "VII. Eye"
        if 60 <= category <= 95:
            return "VIII. Ear"
        logger.debug("Unmapped ICD-10 H-code category", extra={"code": code})
        return UNMAPPED
    if letter == "I":
        return "IX. Circulatory"
    if letter == "J":
        return "X. Respiratory"
    if letter == "K":
        return "XI. Digestive"
    if letter == "L":
        return "XII. Skin"
    if letter == "M":
        return "XIII. Musculoskeletal"
    if letter == "N":
        return "XIV. Genitourinary"
    if letter == "O":
        return "XV. Pregnancy"
    if letter == "P":
        return "XVI. Perinatal"
    if letter == "Q":
        return "XVII. Congenital"
    if letter == "R":
        return "XVIII. Symptoms/signs"
    if letter in {"S", "T"}:
        return "XIX. Injury/poisoning"
    if letter in {"V", "W", "X", "Y"}:
        return "XX. External causes"
    if letter == "Z":
        return "XXI. Factors influencing health"

    logger.debug("Unmapped ICD-10 chapter letter", extra={"code": code})
    return UNMAPPED


def _icd9_chapter(code: str) -> str:
    compact = code.replace(".", "")

    if compact.startswith("V"):
        return "XXI. Factors influencing health"
    if compact.startswith("E"):
        return "XX. External causes"

    match = re.match(r"^(\d{3})", compact)
    if not match:
        logger.debug("Unmapped ICD-9 code format", extra={"code": code})
        return UNMAPPED

    category = int(match.group(1))

    if 1 <= category <= 139:
        return "I. Infectious"
    if 140 <= category <= 239:
        return "II. Neoplasms"
    if 240 <= category <= 279:
        return "IV. Endocrine/metabolic"
    if 280 <= category <= 289:
        return "III. Blood/immune"
    if 290 <= category <= 319:
        return "V. Mental"
    if 320 <= category <= 389:
        return "VI./VII./VIII. Nervous/Eye/Ear"
    if 390 <= category <= 459:
        return "IX. Circulatory"
    if 460 <= category <= 519:
        return "X. Respiratory"
    if 520 <= category <= 579:
        return "XI. Digestive"
    if 580 <= category <= 629:
        return "XIV. Genitourinary"
    if 630 <= category <= 679:
        return "XV. Pregnancy"
    if 680 <= category <= 709:
        return "XII. Skin"
    if 710 <= category <= 739:
        return "XIII. Musculoskeletal"
    if 740 <= category <= 759:
        return "XVII. Congenital"
    if 760 <= category <= 779:
        return "XVI. Perinatal"
    if 780 <= category <= 799:
        return "XVIII. Symptoms/signs"
    if 800 <= category <= 999:
        return "XIX. Injury/poisoning"

    logger.debug("Unmapped ICD-9 category", extra={"code": code})
    return UNMAPPED
