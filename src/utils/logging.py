from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from src import config

_LOG_RECORD_DEFAULT_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


class BudgetExceededError(RuntimeError):
    pass


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _LOG_RECORD_DEFAULT_KEYS and not key.startswith("_")
        }
        payload.update(extra)
        return json.dumps(payload, ensure_ascii=True, default=str)


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(_JSONFormatter())
        logger.addHandler(file_handler)

    return logger


class CostTracker:
    def __init__(
        self,
        run_id: str,
        max_budget_usd: float,
        input_price_per_million: float,
        output_price_per_million: float,
        log_path: Path | None = None,
    ) -> None:
        self.run_id = run_id
        self.max_budget_usd = max_budget_usd
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million
        self.log_path = log_path or (config.LOGS_DIR / "runs" / f"{run_id}_cost.json")

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.n_calls = 0
        self.started_at = time.time()

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._restore_state()

    def _restore_state(self) -> None:
        if not self.log_path.exists():
            return

        last_line = ""
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    last_line = stripped

        if not last_line:
            return

        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:
            return

        if payload.get("run_id") != self.run_id:
            return

        self.total_input_tokens = int(payload.get("total_input_tokens", 0))
        self.total_output_tokens = int(payload.get("total_output_tokens", 0))
        self.total_cost_usd = float(payload.get("total_cost_usd", 0.0))
        self.n_calls = int(payload.get("n_calls", 0))
        self.started_at = float(payload.get("started_at_ts", self.started_at))

    def _append_event(self, *, event: str, input_tokens: int, output_tokens: int) -> None:
        payload = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "event": event,
            "run_id": self.run_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "n_calls": self.n_calls,
            "started_at_ts": self.started_at,
            "max_budget_usd": self.max_budget_usd,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def add_call(self, input_tokens: int, output_tokens: int) -> None:
        input_tokens = int(input_tokens)
        output_tokens = int(output_tokens)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.n_calls += 1

        input_cost = (input_tokens / 1_000_000) * self.input_price_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_million
        self.total_cost_usd += input_cost + output_cost

        self._append_event(event="add_call", input_tokens=input_tokens, output_tokens=output_tokens)

        if self.total_cost_usd > self.max_budget_usd:
            raise BudgetExceededError(
                f"Run '{self.run_id}' exceeded budget: ${self.total_cost_usd:.6f} > "
                f"${self.max_budget_usd:.6f}."
            )

    def summary(self) -> dict[str, float | int | str]:
        elapsed_seconds = time.time() - self.started_at
        budget_remaining = self.max_budget_usd - self.total_cost_usd
        return {
            "run_id": self.run_id,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "n_calls": self.n_calls,
            "elapsed_seconds": elapsed_seconds,
            "budget_remaining": budget_remaining,
        }

    def reset(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.n_calls = 0
        self.started_at = time.time()
        self._append_event(event="reset", input_tokens=0, output_tokens=0)
