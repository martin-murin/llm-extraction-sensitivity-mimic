from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import hashlib

import numpy as np

from src.llm.client import LLMClient


class EmbeddingBackend(ABC):
    """Abstract embedding backend."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return (N, D) float32 L2-normalized embeddings."""
        ...


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """Embedding backend backed by the project's OpenAI client wrapper."""

    def __init__(self, client: LLMClient, model_id: str = "text-embedding-3-small") -> None:
        self._client = client
        self._model_id = model_id
        self._dimension = 3072 if model_id == "text-embedding-3-large" else 1536

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        raw = await self._client.embed(texts, model=self._model_id)
        if raw.size == 0:
            return np.empty((0, self.dimension), dtype=np.float32)

        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normalized = raw / norms
        return normalized.astype(np.float32)


class EmbeddingCache:
    """Simple per-text embedding cache on disk."""

    def __init__(self, cache_dir: Path, backend: EmbeddingBackend) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend

        self.cache_hits = 0
        self.cache_misses = 0
        self.backend_calls = 0

    def _key(self, text: str) -> str:
        hasher = hashlib.sha256()
        hasher.update(self._backend.model_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    def _path_for_key(self, key: str) -> Path:
        return self._dir / f"{key}.npy"

    async def embed_cached(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._backend.dimension), dtype=np.float32)

        results: list[np.ndarray | None] = [None] * len(texts)
        key_to_indices: dict[str, list[int]] = {}
        key_to_text: dict[str, str] = {}

        for index, text in enumerate(texts):
            key = self._key(text)
            path = self._path_for_key(key)
            if path.exists():
                loaded = np.load(path, allow_pickle=False).astype(np.float32)
                if loaded.ndim != 1:
                    loaded = loaded.reshape(-1)
                results[index] = loaded
                self.cache_hits += 1
                continue

            key_to_indices.setdefault(key, []).append(index)
            key_to_text.setdefault(key, text)

        if key_to_text:
            self.cache_misses += len(key_to_text)
            missing_keys = list(key_to_text.keys())
            missing_texts = [key_to_text[key] for key in missing_keys]

            embedded = await self._backend.embed(missing_texts)
            self.backend_calls += 1

            if embedded.shape[0] != len(missing_texts):
                raise RuntimeError(
                    "Embedding backend returned unexpected number of rows: "
                    f"expected {len(missing_texts)}, got {embedded.shape[0]}"
                )

            for row_index, key in enumerate(missing_keys):
                vector = embedded[row_index].astype(np.float32)
                path = self._path_for_key(key)
                np.save(path, vector)
                for original_index in key_to_indices.get(key, []):
                    results[original_index] = vector

        missing_slots = [index for index, value in enumerate(results) if value is None]
        if missing_slots:
            raise RuntimeError(f"Embedding cache assembly failed for indices: {missing_slots}")

        stacked = np.vstack([value for value in results if value is not None])
        return stacked.astype(np.float32)
