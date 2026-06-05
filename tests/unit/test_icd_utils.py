from __future__ import annotations

from src.db.icd_utils import icd10_chapter_from_code


def test_icd10_chapter_from_code_mappings() -> None:
    assert icd10_chapter_from_code("I50.9", 10) == "IX. Circulatory"
    assert icd10_chapter_from_code("4280", 9) == "IX. Circulatory"
    assert icd10_chapter_from_code("V5811", 9) == "XXI. Factors influencing health"
    assert icd10_chapter_from_code("E8497", 9) == "XX. External causes"
    assert icd10_chapter_from_code("", 10) == "ZZ. Unmapped"
    assert icd10_chapter_from_code("??", 10) == "ZZ. Unmapped"
