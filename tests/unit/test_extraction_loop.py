from __future__ import annotations

from pathlib import Path

import asyncio


def test_hf_aki_scripts_use_single_asyncio_run_pattern() -> None:
    hf_src = Path("scripts/75_run_hf_extraction.py").read_text(encoding="utf-8")
    aki_src = Path("scripts/76_run_aki_extraction.py").read_text(encoding="utf-8")
    assert "asyncio.run(_run_all())" in hf_src
    assert "asyncio.run(_run_all())" in aki_src
    assert "outcomes = asyncio.run(_run_batch(batch_ids))" not in hf_src
    assert "outcomes = asyncio.run(_run_batch(batch_ids))" not in aki_src


def test_single_event_loop_multi_batch_pattern_is_safe() -> None:
    async def run_all_batches() -> int:
        total = 0
        sem = asyncio.Semaphore(2)

        async def one_task() -> int:
            async with sem:
                await asyncio.sleep(0)
                return 1

        for _ in range(3):
            outcomes = await asyncio.gather(*[one_task() for _ in range(5)])
            total += sum(outcomes)
        return total

    total = asyncio.run(run_all_batches())
    assert total == 15


def test_resume_pending_selection_from_existing_json() -> None:
    hadm_ids = [101, 102, 103, 104]
    tmp = Path("tests/.tmp_resume_selection")
    tmp.mkdir(parents=True, exist_ok=True)
    for hadm in [101, 103]:
        (tmp / f"{hadm}.json").write_text("{}", encoding="utf-8")
    pending = [h for h in hadm_ids if not (tmp / f"{h}.json").exists()]
    assert pending == [102, 104]
    # cleanup
    for p in tmp.glob("*.json"):
        p.unlink()
    tmp.rmdir()
