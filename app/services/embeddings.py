from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence
from functools import lru_cache
import io
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from app.core.config import Settings, get_settings

_WARMUP_IMAGE_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aF9sAAAAASUVORK5CYII="
)

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


def _import_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingProviderError(
            "Missing dependency 'transformers'. Install backend requirements first."
        ) from exc
    return AutoModel, AutoProcessor


def _import_pil_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingProviderError(
            "Missing dependency 'pillow'. Install backend requirements first."
        ) from exc
    return Image


def _pick_runtime(prefer_bf16: bool) -> tuple[str, Any, str]:
    torch = _import_torch()
    if torch.cuda.is_available():
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if prefer_bf16 and bf16_supported:
            return "cuda", torch.bfloat16, "bf16"
        return "cuda", torch.float16, "fp16"
    return "cpu", torch.float32, "fp32"


def _pool_embeddings(outputs: Any, attention_mask: Any) -> list[list[float]]:
    torch = _import_torch()
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise EmbeddingProviderError("Model output does not contain last_hidden_state.")

    if attention_mask is None:
        pooled = hidden.mean(dim=1)
    else:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.detach().cpu().float().tolist()


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

    # Handle Windows-style absolute paths (e.g., C:\data\image.jpg).
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
        model_kwargs: dict[str, Any] = {}
        if self.device != "cpu":
            model_kwargs["torch_dtype"] = self.dtype

        self.model = SentenceTransformer(
            self.model_id,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            device=self.device,
        )
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
                tokenizer.pad_token = tokenizer.eos_token
        self.model.max_seq_length = self.max_length
        self.embedding_dimension = int(self.model.get_sentence_embedding_dimension())

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
        return vectors.tolist()

    def embed_document(self, text: str) -> list[float]:
        cleaned = (text or "").strip()
        if not cleaned:
            raise EmbeddingProviderError("Cannot generate embedding for empty text.")
        return self.embed_documents([cleaned])[0]


class QwenMultimodalEmbedder:
    _PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

    def __init__(
        self,
        model_id: str,
        *,
        max_length: int,
        prefer_bf16: bool,
        text_batch_size: int,
        image_batch_size: int,
    ) -> None:
        AutoModel, AutoProcessor = _import_transformers()
        self.model_id = model_id
        self.max_length = max(1, max_length)
        self.text_batch_size = max(1, text_batch_size)
        self.image_batch_size = max(1, image_batch_size)
        self.device, self.dtype, self.precision = _pick_runtime(prefer_bf16)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if self.device != "cpu":
            model_kwargs["torch_dtype"] = self.dtype

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            min_pixels=224 * 224,
            max_pixels=512 * 512,
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()

    def _prepare_inputs(
        self,
        *,
        texts: list[str] | None = None,
        images: list[Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"padding": True, "return_tensors": "pt"}

        if texts is not None:
            kwargs["text"] = texts
            if images is None:
                kwargs["truncation"] = True
                kwargs["max_length"] = self.max_length
            else:
                kwargs["truncation"] = False

        if images is not None:
            kwargs["images"] = images

        return self.processor(**kwargs).to(self.device)

    def _build_image_prompt(self, text: str | None) -> str:
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return self._PLACEHOLDER
        if self._PLACEHOLDER in cleaned_text:
            return cleaned_text
        # Qwen3-VL expects image placeholder tokens when image features are present.
        return f"{self._PLACEHOLDER}\n{cleaned_text}"

    def _run_model(self, inputs: Any) -> list[list[float]]:
        torch = _import_torch()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return _pool_embeddings(outputs, inputs.get("attention_mask"))

    def _load_image(self, image_source: str | None) -> Any:
        Image = _import_pil_image()
        local_path = _local_path_from_image_source(image_source)
        if not local_path or not local_path.exists():
            return None
        try:
            with Image.open(local_path) as image:
                return image.convert("RGB")
        except Exception:  # pragma: no cover
            return None

    def _load_image_bytes(self, image_bytes: bytes) -> Any:
        Image = _import_pil_image()
        if not image_bytes:
            return None
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.convert("RGB")
        except Exception:  # pragma: no cover
            return None

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

            inputs = self._prepare_inputs(
                texts=[self._PLACEHOLDER] * len(valid_images),
                images=valid_images,
            )
            vectors.extend(self._run_model(inputs))

        return vectors

    def embed_many_texts(
        self,
        texts: Sequence[str],
        *,
        progress_desc: str | None = None,
    ) -> list[list[float]]:
        cleaned = [(text or "").strip() for text in texts]
        vectors: list[list[float]] = []
        for start, stop in _iter_batches(cleaned, self.text_batch_size, progress_desc):
            batch = cleaned[start:stop]
            inputs = self._prepare_inputs(texts=batch)
            vectors.extend(self._run_model(inputs))
        return vectors

    def embed(
        self,
        *,
        text: str | None = None,
        image_source: str | None = None,
    ) -> list[float]:
        cleaned_text = (text or "").strip()

        if image_source and not cleaned_text:
            vectors = self.embed_many_images([image_source])
            if not vectors:
                raise EmbeddingProviderError(f"Could not load image_source '{image_source}'.")
            return vectors[0]

        if image_source:
            image = self._load_image(image_source)
            if image is None:
                raise EmbeddingProviderError(f"Could not load image_source '{image_source}'.")
            prompt_text = self._build_image_prompt(cleaned_text)
            inputs = self._prepare_inputs(texts=[prompt_text], images=[image])
            return self._run_model(inputs)[0]

        if cleaned_text:
            return self.embed_many_texts([cleaned_text])[0]

        raise EmbeddingProviderError("Provide image_source or text.")

    def embed_image_bytes(self, image_bytes: bytes, *, text: str | None = None) -> list[float]:
        image = self._load_image_bytes(image_bytes)
        if image is None:
            raise EmbeddingProviderError("Could not decode uploaded image bytes.")

        prompt_text = self._build_image_prompt(text)
        inputs = self._prepare_inputs(texts=[prompt_text], images=[image])
        return self._run_model(inputs)[0]

    def embed_many_image_bytes(
        self,
        image_bytes_values: Sequence[bytes],
        *,
        text: str | None = None,
        progress_desc: str | None = None,
    ) -> list[list[float]]:
        if not image_bytes_values:
            return []

        values = list(image_bytes_values)
        vectors: list[list[float]] = []
        prompt_text = self._build_image_prompt(text)

        for start, stop in _iter_batches(values, self.image_batch_size, progress_desc):
            batch_values = values[start:stop]
            images = [self._load_image_bytes(image_bytes) for image_bytes in batch_values]
            valid_images = [img for img in images if img is not None]
            if not valid_images:
                continue

            inputs = self._prepare_inputs(
                texts=[prompt_text] * len(valid_images),
                images=valid_images,
            )
            vectors.extend(self._run_model(inputs))

        return vectors


class EmbeddingProvider:
    """Embedding provider aligned with ../HBIM/Indexer behavior."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.text_model_id = settings.text_embedding_model_resolved
        self.multimodal_model_id = settings.multimodal_embedding_model_resolved

        self._text_embedder: QwenTextEmbedder | None = None
        self._multimodal_embedder: QwenMultimodalEmbedder | None = None

    def _get_text_embedder(self) -> QwenTextEmbedder:
        if self._text_embedder is None:
            self._text_embedder = QwenTextEmbedder(
                self.text_model_id,
                max_length=self.settings.EMBEDDING_MAX_LENGTH,
                prefer_bf16=self.settings.EMBEDDING_PREFER_BF16,
                batch_size=self.settings.TEXT_EMBEDDING_BATCH_SIZE,
            )
        return self._text_embedder

    def _get_multimodal_embedder(self) -> QwenMultimodalEmbedder:
        if self._multimodal_embedder is None:
            self._multimodal_embedder = QwenMultimodalEmbedder(
                self.multimodal_model_id,
                max_length=self.settings.EMBEDDING_MAX_LENGTH,
                prefer_bf16=self.settings.EMBEDDING_PREFER_BF16,
                text_batch_size=self.settings.MULTIMODAL_TEXT_EMBEDDING_BATCH_SIZE,
                image_batch_size=self.settings.MULTIMODAL_IMAGE_EMBEDDING_BATCH_SIZE,
            )
        return self._multimodal_embedder

    async def embed_text(self, text: str) -> list[float]:
        embedder = self._get_text_embedder()
        return await asyncio.to_thread(embedder.embed_document, text)

    async def embed_multimodal(self, *, text: str | None = None, image_url: str | None = None) -> list[float]:
        embedder = self._get_multimodal_embedder()
        return await asyncio.to_thread(
            embedder.embed,
            text=text,
            image_source=image_url,
        )

    async def embed_multimodal_image_bytes(
        self,
        *,
        image_bytes: bytes,
        text: str | None = None,
    ) -> list[float]:
        embedder = self._get_multimodal_embedder()
        return await asyncio.to_thread(
            embedder.embed_image_bytes,
            image_bytes,
            text=text,
        )

    async def embed_many_multimodal_image_bytes(
        self,
        *,
        image_bytes_values: Sequence[bytes],
        text: str | None = None,
    ) -> list[list[float]]:
        embedder = self._get_multimodal_embedder()
        return await asyncio.to_thread(
            embedder.embed_many_image_bytes,
            image_bytes_values,
            text=text,
        )

    async def warmup(self, *, include_multimodal: bool = True) -> None:
        await self.embed_text("warmup")
        if include_multimodal:
            await self.embed_multimodal_image_bytes(
                image_bytes=_WARMUP_IMAGE_BYTES,
                text="warmup",
            )


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider(get_settings())
