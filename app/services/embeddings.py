from __future__ import annotations

import asyncio
from collections.abc import Sequence
from functools import lru_cache
import io
import json
import logging
import math
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from app.core.config import Settings, get_settings
from app.core.logging import log_event

logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Sequence[Any], **kwargs: Any) -> Sequence[Any]:  # type: ignore[no-redef]
        return iterable


class EmbeddingProviderError(RuntimeError):
    """Raised when embedding generation is unavailable or fails."""


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingProviderError(
            "Missing dependency 'torch'. Install backend requirements first."
        ) from exc
    return torch


def _import_sentence_transformers() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingProviderError(
            "Missing dependency 'sentence-transformers'. Install backend requirements first."
        ) from exc
    return SentenceTransformer


def _import_flag_embedding() -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingProviderError(
            "Missing dependency 'FlagEmbedding'. Install backend requirements first."
        ) from exc
    return BGEM3FlagModel


def _import_pil_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingProviderError(
            "Missing dependency 'pillow'. Install backend requirements first."
        ) from exc
    return Image


def _vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _l2_normalize_vector(values: Sequence[float]) -> list[float]:
    norm = _vector_norm(values)
    if norm <= 0:
        return [float(value) for value in values]
    return [float(value) / norm for value in values]


def _pick_runtime(prefer_bf16: bool) -> tuple[str, Any, str]:
    torch = _import_torch()
    if torch.cuda.is_available():
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if prefer_bf16 and bf16_supported:
            return "cuda", torch.bfloat16, "bf16"
        return "cuda", torch.float16, "fp16"
    return "cpu", torch.float32, "fp32"


def _iter_batches(
    values: Sequence[Any],
    batch_size: int,
    progress_desc: str | None,
) -> Sequence[tuple[int, int]]:
    if batch_size <= 0:
        raise EmbeddingProviderError("batch_size must be greater than zero.")
    desc = progress_desc or "Generating embeddings"
    return (
        (start, min(start + batch_size, len(values)))
        for start in tqdm(range(0, len(values), batch_size), desc=desc, unit="batch")
    )


def _local_path_from_image_source(image_source: str | None) -> Path | None:
    raw = (image_source or "").strip()
    if not raw:
        return None

    if len(raw) >= 2 and raw[1] == ":":
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else candidate.resolve()

    if raw.startswith("file://"):
        parsed = urlparse(raw)
        path_value = unquote(parsed.path)
        if parsed.netloc:
            path_value = f"//{parsed.netloc}{path_value}"
    else:
        parsed = urlparse(raw)
        if parsed.scheme and parsed.scheme not in {"file"}:
            return None
        path_value = raw

    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()

    return candidate


class OpenRouterEmbeddingsClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model_id: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise EmbeddingProviderError(
                "USE_OPENROUTER_BGE_M3=true requires OPENROUTER_API_KEY to be set."
            )

        self.base_url = base_url.strip().rstrip("/")
        if not self.base_url:
            raise EmbeddingProviderError(
                "OPENROUTER_BASE_URL cannot be empty when USE_OPENROUTER_BGE_M3=true."
            )

        self.model_id = model_id.strip()
        if not self.model_id:
            raise EmbeddingProviderError(
                "OPENROUTER_BGE_MODEL cannot be empty when USE_OPENROUTER_BGE_M3=true."
            )

        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def _request_embeddings(self, inputs: Sequence[str]) -> list[list[float]]:
        body = json.dumps(
            {
                "model": self.model_id,
                "input": list(inputs),
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/embeddings",
            data=body,
            method="POST",
        )
        request.add_header("Authorization", f"Bearer {self.api_key}")
        request.add_header("Content-Type", "application/json")

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status_code = int(getattr(response, "status", 200))
                raw = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingProviderError(
                f"OpenRouter embeddings request failed with HTTP {exc.code}: {details}"
            ) from exc
        except URLError as exc:
            raise EmbeddingProviderError(
                f"OpenRouter embeddings request failed: {exc.reason}"
            ) from exc
        except Exception as exc:
            raise EmbeddingProviderError(
                f"OpenRouter embeddings request failed unexpectedly: {exc}"
            ) from exc

        if status_code >= 400:
            raise EmbeddingProviderError(
                f"OpenRouter embeddings request failed with HTTP {status_code}: {raw}"
            )

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EmbeddingProviderError(
                f"OpenRouter embeddings returned invalid JSON: {raw[:240]}"
            ) from exc

        if not isinstance(payload, dict):
            raise EmbeddingProviderError(
                "OpenRouter embeddings returned an invalid payload: expected object with data list."
            )
        data = payload.get("data")
        if not isinstance(data, list):
            raise EmbeddingProviderError(
                "OpenRouter embeddings response missing 'data' list."
            )

        vectors_by_index: dict[int, list[float]] = {}
        expected = len(inputs)
        for item in data:
            if not isinstance(item, dict):
                raise EmbeddingProviderError(
                    "OpenRouter embeddings response contains an invalid item."
                )
            index = item.get("index")
            vector = item.get("embedding")
            if not isinstance(index, int) or index < 0 or index >= expected:
                raise EmbeddingProviderError(
                    f"OpenRouter embeddings returned invalid index: {index!r}."
                )
            if index in vectors_by_index:
                raise EmbeddingProviderError(
                    f"OpenRouter embeddings returned duplicate index: {index}."
                )
            if not isinstance(vector, list):
                raise EmbeddingProviderError(
                    f"OpenRouter embeddings item at index {index} is missing a valid 'embedding' list."
                )
            try:
                vectors_by_index[index] = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise EmbeddingProviderError(
                    f"OpenRouter embeddings item at index {index} contains non-numeric values."
                ) from exc

        if len(vectors_by_index) != expected:
            raise EmbeddingProviderError(
                f"OpenRouter embeddings returned {len(vectors_by_index)} vectors for {expected} inputs."
            )

        return [vectors_by_index[idx] for idx in range(expected)]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [(text or "").strip() for text in texts]
        if not cleaned:
            return []
        return self._request_embeddings(cleaned)


class QwenTextEmbedder:
    def __init__(
        self,
        model_id: str,
        *,
        max_length: int,
        prefer_bf16: bool,
        batch_size: int,
    ) -> None:
        SentenceTransformer = _import_sentence_transformers()
        self.model_id = model_id
        self.max_length = max(1, max_length)
        self.batch_size = max(1, batch_size)
        self.device, self.dtype, self.precision = _pick_runtime(prefer_bf16)


        self.model = SentenceTransformer(
            self.model_id,
            trust_remote_code=True,
            model_kwargs={"dtype": self.dtype},
            device=self.device,
        )
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
                tokenizer.pad_token = tokenizer.eos_token
        self.model.max_seq_length = self.max_length
        self.embedding_dimension = int(self.model.get_embedding_dimension())

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        progress_desc: str | None = None,
    ) -> list[list[float]]:
        cleaned = [(text or "").strip() for text in texts]
        if not cleaned:
            return []

        vectors = self.model.encode(
            cleaned,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress_desc is not None,
        )
        payload = vectors.tolist()
        return payload

    def embed_document(self, text: str) -> list[float]:
        cleaned = (text or "").strip()
        if not cleaned:
            raise EmbeddingProviderError("Cannot generate embedding for empty text.")
        return self.embed_documents([cleaned])[0]


class BGEM3TextEmbedder:
    def __init__(
        self,
        model_id: str,
        *,
        max_length: int,
        batch_size: int,
        expected_dimension: int,
    ) -> None:
        BGEM3FlagModel = _import_flag_embedding()
        self.model_id = model_id
        self.max_length = max(1, max_length)
        self.batch_size = max(1, batch_size)
        self.expected_dimension = max(1, expected_dimension)

        use_fp16 = False
        try:
            torch = _import_torch()
            use_fp16 = bool(torch.cuda.is_available())
        except EmbeddingProviderError:
            use_fp16 = False

        self.model = BGEM3FlagModel(self.model_id, use_fp16=use_fp16)
        self.embedding_dimension = self.expected_dimension

    def _normalize_payload(self, payload: Any) -> list[list[float]]:
        if hasattr(payload, "tolist"):
            payload = payload.tolist()
        if payload is None:
            return []
        if isinstance(payload, list) and payload and isinstance(payload[0], (int, float)):
            payload = [payload]
        if not isinstance(payload, list):
            return []

        normalized: list[list[float]] = []
        for vector in payload:
            if not isinstance(vector, (list, tuple)):
                raise EmbeddingProviderError("BGEM3 returned an invalid dense vector payload.")
            casted = [float(value) for value in vector]
            if len(casted) != self.embedding_dimension:
                raise EmbeddingProviderError(
                    f"Unexpected BGEM3 embedding dimension: got {len(casted)}, expected {self.embedding_dimension}."
                )
            normalized.append(_l2_normalize_vector(casted))
        return normalized

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        progress_desc: str | None = None,
    ) -> list[list[float]]:
        del progress_desc
        cleaned = [(text or "").strip() for text in texts]
        if not cleaned:
            return []

        result = self.model.encode(
            cleaned,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = result.get("dense_vecs") if isinstance(result, dict) else None
        if dense is None:
            raise EmbeddingProviderError("BGEM3 encode did not return dense_vecs.")

        vectors = self._normalize_payload(dense)
        return vectors

    def embed_document(self, text: str) -> list[float]:
        cleaned = (text or "").strip()
        if not cleaned:
            raise EmbeddingProviderError("Cannot generate embedding for empty text.")
        return self.embed_documents([cleaned])[0]


class OpenRouterBGETextEmbedder:
    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None,
        base_url: str,
        batch_size: int,
        expected_dimension: int,
    ) -> None:
        self.model_id = model_id.strip()
        self.batch_size = max(1, batch_size)
        self.expected_dimension = max(1, expected_dimension)
        self.client = OpenRouterEmbeddingsClient(
            api_key=api_key,
            base_url=base_url,
            model_id=self.model_id,
        )
        self.embedding_dimension = self.expected_dimension

    def _normalize_payload(self, payload: Any) -> list[list[float]]:
        if not isinstance(payload, list):
            raise EmbeddingProviderError("OpenRouter embeddings returned an invalid vectors payload.")
        normalized: list[list[float]] = []
        for vector in payload:
            if not isinstance(vector, list):
                raise EmbeddingProviderError("OpenRouter embeddings returned an invalid vector entry.")
            casted = [float(value) for value in vector]
            if len(casted) != self.embedding_dimension:
                raise EmbeddingProviderError(
                    f"Unexpected OpenRouter embedding dimension: got {len(casted)}, expected {self.embedding_dimension}."
                )
            normalized.append(_l2_normalize_vector(casted))
        return normalized

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        progress_desc: str | None = None,
    ) -> list[list[float]]:
        del progress_desc
        cleaned = [(text or "").strip() for text in texts]
        if not cleaned:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            raw_vectors = self.client.embed_texts(batch)
            vectors.extend(self._normalize_payload(raw_vectors))

        return vectors

    def embed_document(self, text: str) -> list[float]:
        cleaned = (text or "").strip()
        if not cleaned:
            raise EmbeddingProviderError("Cannot generate embedding for empty text.")
        return self.embed_documents([cleaned])[0]


class QwenMultimodalImageEmbedder:
    def __init__(
        self,
        model_id: str,
        *,
        max_length: int,
        prefer_bf16: bool,
        image_batch_size: int,
    ) -> None:
        SentenceTransformer = _import_sentence_transformers()
        self.model_id = model_id
        self.max_length = max(1, max_length)
        self.image_batch_size = max(1, image_batch_size)
        self.device, self.dtype, self.precision = _pick_runtime(prefer_bf16)


        self.model = SentenceTransformer(
            self.model_id,
            trust_remote_code=True,
            model_kwargs={"dtype": self.dtype},
            device=self.device,
        )
        self.model.max_seq_length = self.max_length
        self.embedding_dimension = int(self.model.get_embedding_dimension())

    def _load_image(self, image_source: str | None) -> Any:
        Image = _import_pil_image()
        local_path = _local_path_from_image_source(image_source)
        if not local_path or not local_path.exists():
            return None
        try:
            with Image.open(local_path) as image:
                return image.convert("RGB").copy()
        except Exception:  # pragma: no cover
            return None

    def _load_image_bytes(self, image_bytes: bytes) -> Any:
        Image = _import_pil_image()
        if not image_bytes:
            raise EmbeddingProviderError("Could not decode uploaded image bytes.")
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB").copy()



    def embed_many_images(
        self,
        image_sources: Sequence[str],
        *,
        progress_desc: str | None = None,
    ) -> list[list[float]]:
        if not image_sources:
            return []

        values = list(image_sources)
        vectors: list[list[float]] = []

        for start, stop in _iter_batches(values, self.image_batch_size, progress_desc):
            batch_sources = values[start:stop]
            images = [self._load_image(src) for src in batch_sources]
            valid_images = [img for img in images if img is not None]
            if not valid_images:
                continue

            encoded = self.model.encode(
                valid_images,
                batch_size=len(valid_images),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            payload = encoded.tolist()
            vectors.extend(payload)

        return vectors

    def embed(self, *, text: str | None = None, image_source: str | None = None) -> list[float]:
        del text
        if image_source is None:
            raise EmbeddingProviderError("Provide image_source for multimodal embedding.")
        vectors = self.embed_many_images([image_source])
        if not vectors:
            raise EmbeddingProviderError(f"Could not load image_source '{image_source}'.")
        return vectors[0]

    def embed_image_bytes(self, image_bytes: bytes, *, text: str | None = None) -> list[float]:
        del text
        vectors = self.embed_many_image_bytes([image_bytes], text=None)
        if not vectors:
            raise EmbeddingProviderError("Could not decode uploaded image bytes.")
        return vectors[0]

    def embed_many_image_bytes(
        self,
        image_bytes_values: Sequence[bytes],
        *,
        text: str | None = None,
        progress_desc: str | None = None,
    ) -> list[list[float]]:
        del text
        if not image_bytes_values:
            return []

        values = list(image_bytes_values)
        vectors: list[list[float]] = []

        for start, stop in _iter_batches(values, self.image_batch_size, progress_desc):
            batch_values = values[start:stop]
            valid_images: list[Any] = []
            for image_bytes in batch_values:
                try:
                    valid_images.append(self._load_image_bytes(image_bytes))
                except Exception:  # pragma: no cover
                    continue

            if not valid_images:
                continue

            encoded = self.model.encode(
                valid_images,
                batch_size=len(valid_images),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            payload = encoded.tolist()
            vectors.extend(payload)

        return vectors


class EmbeddingProvider:
    """Embedding provider aligned with ../HBIM/Indexer behavior."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.text_model_id = settings.text_embedding_model_resolved
        self.multimodal_model_id = settings.multimodal_embedding_model_resolved

        self._text_embedder: QwenTextEmbedder | BGEM3TextEmbedder | OpenRouterBGETextEmbedder | None = None
        self._multimodal_embedder: QwenMultimodalImageEmbedder | None = None


    def _get_text_embedder(self) -> QwenTextEmbedder | BGEM3TextEmbedder | OpenRouterBGETextEmbedder:
        if self._text_embedder is None:
            started_at = time.perf_counter()
            log_event(
                logger,
                logging.INFO,
                "embedding.text_model.load.start",
                model=self.text_model_id,
                openrouter=self.settings.USE_OPENROUTER_BGE_M3,
            )
            normalized_model_id = self.text_model_id.strip().lower()
            if normalized_model_id == "baai/bge-m3":
                if self.settings.USE_OPENROUTER_BGE_M3:
                    self._text_embedder = OpenRouterBGETextEmbedder(
                        self.settings.openrouter_bge_model_resolved,
                        api_key=self.settings.OPENROUTER_API_KEY,
                        base_url=self.settings.openrouter_base_url_resolved,
                        batch_size=self.settings.TEXT_EMBEDDING_BATCH_SIZE,
                        expected_dimension=self.settings.ARTIFACT_TEXT_EMBEDDING_DIMENSION,
                    )
                else:
                    self._text_embedder = BGEM3TextEmbedder(
                        self.text_model_id,
                        max_length=self.settings.EMBEDDING_MAX_LENGTH,
                        batch_size=self.settings.TEXT_EMBEDDING_BATCH_SIZE,
                        expected_dimension=self.settings.ARTIFACT_TEXT_EMBEDDING_DIMENSION,
                    )
            else:
                self._text_embedder = QwenTextEmbedder(
                    self.text_model_id,
                    max_length=self.settings.EMBEDDING_MAX_LENGTH,
                    prefer_bf16=self.settings.EMBEDDING_PREFER_BF16,
                    batch_size=self.settings.TEXT_EMBEDDING_BATCH_SIZE,
                )
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.INFO,
                "embedding.text_model.load.finish",
                model=self.text_model_id,
                dimension=getattr(self._text_embedder, "embedding_dimension", None),
                duration_ms=round(duration_ms, 1),
            )
        return self._text_embedder

    def _get_multimodal_embedder(self) -> QwenMultimodalImageEmbedder:
        if self._multimodal_embedder is None:
            started_at = time.perf_counter()
            log_event(
                logger,
                logging.INFO,
                "embedding.multimodal_model.load.start",
                model=self.multimodal_model_id,
            )
            self._multimodal_embedder = QwenMultimodalImageEmbedder(
                self.multimodal_model_id,
                max_length=self.settings.EMBEDDING_MAX_LENGTH,
                prefer_bf16=self.settings.EMBEDDING_PREFER_BF16,
                image_batch_size=self.settings.MULTIMODAL_IMAGE_EMBEDDING_BATCH_SIZE,
            )
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.INFO,
                "embedding.multimodal_model.load.finish",
                model=self.multimodal_model_id,
                dimension=getattr(self._multimodal_embedder, "embedding_dimension", None),
                duration_ms=round(duration_ms, 1),
            )
        return self._multimodal_embedder

    async def embed_text(self, text: str) -> list[float]:
        started_at = time.perf_counter()
        embedder = self._get_text_embedder()
        log_event(
            logger,
            logging.INFO,
            "embedding.text.start",
            model=getattr(embedder, "model_id", self.text_model_id),
            chars=len((text or "").strip()),
        )
        try:
            vector = await asyncio.to_thread(embedder.embed_document, text)
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "embedding.text.error",
                duration_ms=round(duration_ms, 1),
                error=exc,
            )
            raise
        duration_ms = (time.perf_counter() - started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "embedding.text.finish",
            dimension=len(vector),
            duration_ms=round(duration_ms, 1),
        )
        return vector

    async def embed_multimodal(self, *, text: str | None = None, image_url: str | None = None) -> list[float]:
        started_at = time.perf_counter()
        embedder = self._get_multimodal_embedder()
        log_event(
            logger,
            logging.INFO,
            "embedding.multimodal.start",
            model=getattr(embedder, "model_id", self.multimodal_model_id),
            has_text=bool((text or "").strip()),
            has_image_url=bool((image_url or "").strip()),
        )
        try:
            vector = await asyncio.to_thread(
                embedder.embed,
                text=text,
                image_source=image_url,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "embedding.multimodal.error",
                duration_ms=round(duration_ms, 1),
                error=exc,
            )
            raise
        duration_ms = (time.perf_counter() - started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "embedding.multimodal.finish",
            dimension=len(vector),
            duration_ms=round(duration_ms, 1),
        )
        return vector

    async def embed_multimodal_image_bytes(
        self,
        *,
        image_bytes: bytes,
        text: str | None = None,
    ) -> list[float]:
        started_at = time.perf_counter()
        embedder = self._get_multimodal_embedder()
        log_event(
            logger,
            logging.INFO,
            "embedding.image_bytes.start",
            model=getattr(embedder, "model_id", self.multimodal_model_id),
            bytes=len(image_bytes or b""),
            has_text=bool((text or "").strip()),
        )
        try:
            vector = await asyncio.to_thread(
                embedder.embed_image_bytes,
                image_bytes,
                text=text,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "embedding.image_bytes.error",
                duration_ms=round(duration_ms, 1),
                error=exc,
            )
            raise
        duration_ms = (time.perf_counter() - started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "embedding.image_bytes.finish",
            dimension=len(vector),
            duration_ms=round(duration_ms, 1),
        )
        return vector

    async def embed_many_multimodal_image_bytes(
        self,
        *,
        image_bytes_values: Sequence[bytes],
        text: str | None = None,
    ) -> list[list[float]]:
        started_at = time.perf_counter()
        embedder = self._get_multimodal_embedder()
        image_count = len(image_bytes_values)
        log_event(
            logger,
            logging.INFO,
            "embedding.image_bytes_batch.start",
            model=getattr(embedder, "model_id", self.multimodal_model_id),
            image_count=image_count,
            total_bytes=sum(len(value or b"") for value in image_bytes_values),
            has_text=bool((text or "").strip()),
        )
        try:
            vectors = await asyncio.to_thread(
                embedder.embed_many_image_bytes,
                image_bytes_values,
                text=text,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "embedding.image_bytes_batch.error",
                image_count=image_count,
                duration_ms=round(duration_ms, 1),
                error=exc,
            )
            raise
        duration_ms = (time.perf_counter() - started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "embedding.image_bytes_batch.finish",
            image_count=image_count,
            vector_count=len(vectors),
            duration_ms=round(duration_ms, 1),
        )
        return vectors

    def preload_models(self) -> None:
        # Eager-load both embedders at server startup to avoid first-request latency.
        self._get_text_embedder()
        self._get_multimodal_embedder()

    async def warmup(self, *, include_multimodal: bool = True) -> None:
        started_at = time.perf_counter()
        log_event(
            logger,
            logging.INFO,
            "embedding.warmup.start",
            include_multimodal=include_multimodal,
        )

        def _warmup_sync() -> None:
            self._get_text_embedder()
            if include_multimodal:
                self._get_multimodal_embedder()

        await asyncio.to_thread(_warmup_sync)
        duration_ms = (time.perf_counter() - started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "embedding.warmup.finish",
            include_multimodal=include_multimodal,
            duration_ms=round(duration_ms, 1),
        )


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider(get_settings())
