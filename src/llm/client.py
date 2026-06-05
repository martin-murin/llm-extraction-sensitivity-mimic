from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

import numpy as np
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src import config
from src.utils.logging import CostTracker

T = TypeVar("T")
RETRY_WAIT_INITIAL_SECONDS = 1.0
RETRY_WAIT_MAX_SECONDS = 30.0
RETRY_EXP_BASE = 2.0
RETRY_JITTER_SECONDS = 2.0


def retry_policy_label() -> str:
    return (
        f"stop_after_attempt({config.MAX_RETRIES}), "
        "wait_exponential_jitter(initial=1,max=30,exp_base=2,jitter=2), "
        "retry_on=[429, APITimeoutError, APIConnectionError]"
    )


def is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    return getattr(exc, "status_code", None) == 429


def build_retrying() -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential_jitter(
            initial=RETRY_WAIT_INITIAL_SECONDS,
            max=RETRY_WAIT_MAX_SECONDS,
            exp_base=RETRY_EXP_BASE,
            jitter=RETRY_JITTER_SECONDS,
        ),
        retry=retry_if_exception(is_retryable_exception),
        reraise=True,
    )


class LLMClient:
    def __init__(
        self,
        semaphore_limit: int = config.MAX_CONCURRENT_REQUESTS,
        *,
        run_id: str = "default",
        max_budget_usd: float = float("inf"),
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
    ) -> None:
        self.client = AsyncOpenAI(api_key=config.SETTINGS.openai_api_key)
        self.semaphore_limit = int(semaphore_limit)
        self.semaphore = asyncio.Semaphore(semaphore_limit)

        self.cost_tracker = CostTracker(
            run_id=run_id,
            max_budget_usd=max_budget_usd,
            input_price_per_million=input_price_per_million or config.INPUT_PRICE_PER_MILLION_USD,
            output_price_per_million=(
                output_price_per_million or config.OUTPUT_PRICE_PER_MILLION_USD
            ),
            log_path=config.LOGS_DIR / "runs" / f"{run_id}_cost.json",
        )

    async def _with_retry(self, fn: Callable[..., Awaitable[T]], **kwargs: Any) -> T:
        async for attempt in build_retrying():
            with attempt:
                async with self.semaphore:
                    return await fn(**kwargs)

        raise RuntimeError("Retry loop exited without returning a value.")

    def _track_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.cost_tracker.add_call(input_tokens=prompt_tokens, output_tokens=completion_tokens)

    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        response_format: Any | None = None,
        max_completion_tokens: int | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> ChatCompletion:
        payload: dict[str, Any] = {
            "model": model or config.MODEL_ID,
            "messages": list(messages),
            "temperature": config.TEMPERATURE if temperature is None else temperature,
            "max_completion_tokens": max_completion_tokens or config.MAX_COMPLETION_TOKENS,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        response = await self._with_retry(self.client.chat.completions.create, **payload)

        usage = response.usage
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self._track_usage(prompt_tokens, completion_tokens)

        return response

    async def embed(self, texts: list[str], model: str | None = None) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        response = await self._with_retry(
            self.client.embeddings.create,
            model=model or config.EMBEDDING_MODEL_ID,
            input=texts,
        )

        usage = response.usage
        if usage is not None:
            prompt_tokens = int(
                getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0)
            )
            self._track_usage(prompt_tokens, 0)

        embeddings = [item.embedding for item in response.data]
        return np.asarray(embeddings, dtype=np.float32)
