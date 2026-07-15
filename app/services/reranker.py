from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings


class RerankerError(RuntimeError):
    """Raised when reranking is unavailable or fails."""


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RerankerError(
            "Missing dependency 'torch'. Install backend requirements first."
        ) from exc
    return torch


def _import_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RerankerError(
            "Missing dependency 'transformers'. Install backend requirements first."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


class PairwiseRerankerService:
    """Second-stage reranker using Qwen recommended pairwise scoring."""

    _PROMPT_TEMPLATE = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    _SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query and the "
        "Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    )
    _PREFIX = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    _YES_TOKEN = "yes"
    _NO_TOKEN = "no"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.reranker_model_resolved
        self.revision = settings.reranker_model_revision_resolved
        self.instruction = settings.RERANKER_INSTRUCTION.strip()
        self.max_length = max(int(settings.RERANKER_MAX_LENGTH), 32)
        self.batch_size = max(int(settings.RERANKER_BATCH_SIZE), 1)
        self.prefer_bf16 = bool(settings.RERANKER_PREFER_BF16)
        self.device: str | None = None
        self.dtype: Any | None = None
        self.score_yes_token_id: int | None = None
        self.score_no_token_id: int | None = None
        self.prefix_tokens: list[int] | None = None
        self.suffix_tokens: list[int] | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _resolve_runtime(self) -> tuple[str, Any]:
        torch = _import_torch()
        if torch.cuda.is_available():
            bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            if self.prefer_bf16 and bf16_supported:
                return "cuda", torch.bfloat16
            return "cuda", torch.float16
        return "cpu", torch.float32

    def _get_tokenizer_and_model(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        AutoModelForCausalLM, AutoTokenizer = _import_transformers()
        self.device, self.dtype = self._resolve_runtime()

        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "padding_side": "left",
        }
        if self.revision:
            tokenizer_kwargs["revision"] = self.revision
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
        if tokenizer.pad_token is None and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if self.revision:
            model_kwargs["revision"] = self.revision
        if self.device != "cpu":
            # transformers >= 5 renamed torch_dtype -> dtype.
            model_kwargs["dtype"] = self.dtype
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        except Exception as exc:
            raise RerankerError(f"Failed to load reranker model '{self.model_id}': {exc}") from exc

        model.eval()
        if self.device:
            model = model.to(self.device)

        yes_ids = tokenizer(
            self._YES_TOKEN,
            add_special_tokens=False,
            return_attention_mask=False,
        ).get("input_ids", [])
        no_ids = tokenizer(
            self._NO_TOKEN,
            add_special_tokens=False,
            return_attention_mask=False,
        ).get("input_ids", [])
        if not yes_ids or not no_ids:
            raise RerankerError("Could not resolve yes/no token ids for reranker scoring.")
        self.score_yes_token_id = int(yes_ids[0])
        self.score_no_token_id = int(no_ids[0])

        prefix_text = self._PREFIX.format(
            system=self._SYSTEM_PROMPT,
        )
        self.prefix_tokens = tokenizer(
            prefix_text,
            add_special_tokens=False,
            return_attention_mask=False,
        ).get("input_ids", [])
        self.suffix_tokens = tokenizer(
            self._SUFFIX,
            add_special_tokens=False,
            return_attention_mask=False,
        ).get("input_ids", [])
        if self.suffix_tokens is None:
            self.suffix_tokens = []

        self._tokenizer = tokenizer
        self._model = model
        return self._tokenizer, self._model

    def _build_document_text(self, doc: dict[str, Any]) -> str:
        title = str(doc.get("title") or "").strip()
        inventory = str(doc.get("inventory") or "").strip()
        snippet = str(doc.get("snippet") or "").strip()
        description = str(doc.get("description") or "").strip()
        category = str(doc.get("category") or "").strip()
        location = str(doc.get("location") or "").strip()
        museum_name = str(doc.get("museum_name") or "").strip()

        year_bits: list[str] = []
        if doc.get("initial_year") is not None:
            year_bits.append(f"initial_year={doc.get('initial_year')}")
        if doc.get("final_year") is not None:
            year_bits.append(f"final_year={doc.get('final_year')}")

        parts = [part for part in [title, snippet, description] if part]
        if inventory:
            parts.append(f"inventory: {inventory}")
        if category:
            parts.append(f"category: {category}")
        if location:
            parts.append(f"location: {location}")
        if museum_name:
            parts.append(f"museum: {museum_name}")
        if year_bits:
            parts.append(", ".join(year_bits))

        text = "\n".join(parts).strip()
        return text or "no-document-text"

    def _score_batch(self, *, query_text: str, batch_docs: list[dict[str, Any]]) -> list[float]:
        torch = _import_torch()
        tokenizer, model = self._get_tokenizer_and_model()
        if self.score_yes_token_id is None or self.score_no_token_id is None:
            raise RerankerError("Reranker scorer tokens are not initialized.")

        if self.prefix_tokens is None or self.suffix_tokens is None:
            raise RerankerError("Reranker prompt tokens are not initialized.")

        pair_texts: list[str] = []
        for doc in batch_docs:
            pair_texts.append(
                self._PROMPT_TEMPLATE.format(
                    instruction=self.instruction,
                    query=query_text.strip() or "empty-query",
                    document=self._build_document_text(doc),
                )
            )

        reserve_tokens = len(self.prefix_tokens) + len(self.suffix_tokens)
        max_payload_len = max(self.max_length - reserve_tokens, 16)

        tokenized = tokenizer(
            pair_texts,
            padding=False,
            truncation=True,
            max_length=max_payload_len,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        input_ids_batch = tokenized.get("input_ids", [])
        if not isinstance(input_ids_batch, list) or not input_ids_batch:
            raise RerankerError("Reranker tokenizer returned no input ids.")

        merged_input_ids: list[list[int]] = []
        for ids in input_ids_batch:
            merged = [*self.prefix_tokens, *ids, *self.suffix_tokens]
            if not merged:
                merged = list(self.suffix_tokens)
            merged_input_ids.append(merged)

        padded = tokenizer.pad(
            {"input_ids": merged_input_ids},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if hasattr(padded, "to") and self.device:
            padded = padded.to(self.device)

        with torch.no_grad():
            logits = model(**padded).logits[:, -1, :]

        yes_logits = logits[:, self.score_yes_token_id]
        no_logits = logits[:, self.score_no_token_id]
        stacked = torch.stack([no_logits, yes_logits], dim=1)
        scores = torch.nn.functional.softmax(stacked, dim=1)[:, 1]
        values = scores.detach().cpu().float().tolist()
        return [float(value) for value in values]

    def _rerank_sync(
        self,
        *,
        query_text: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        if not query_text.strip():
            return [dict(doc) for doc in documents[:top_k]]

        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            stop = min(start + self.batch_size, len(documents))
            batch_docs = documents[start:stop]
            try:
                batch_scores = self._score_batch(query_text=query_text, batch_docs=batch_docs)
            except Exception as exc:
                raise RerankerError(f"Failed to score query-document pairs: {exc}") from exc
            scores.extend(batch_scores)

        if len(scores) != len(documents):
            raise RerankerError(
                f"Unexpected reranker output size. expected={len(documents)} got={len(scores)}"
            )

        scored_docs: list[dict[str, Any]] = []
        for index, doc in enumerate(documents):
            score_value = scores[index]
            scored_doc = dict(doc)
            scored_doc["rerank_score"] = float(score_value)
            scored_doc["retrieval_rank"] = index + 1
            scored_docs.append(scored_doc)

        scored_docs.sort(key=lambda item: item.get("rerank_score", float("-inf")), reverse=True)
        return scored_docs[:top_k]

    async def rerank(
        self,
        *,
        query_text: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        effective_top_k = max(int(top_k), 1)
        if not self.settings.CHAT_ENABLE_RERANKING:
            return [dict(doc) for doc in documents[:effective_top_k]]
        return await asyncio.to_thread(
            self._rerank_sync,
            query_text=query_text,
            documents=documents,
            top_k=effective_top_k,
        )

    async def warmup(self) -> None:
        if not self.settings.CHAT_ENABLE_RERANKING:
            return
        await self.rerank(
            query_text="warmup",
            documents=[
                {
                    "title": "warmup",
                    "snippet": "warmup",
                    "description": "warmup",
                    "artifact_id": "warmup_artifact",
                }
            ],
            top_k=1,
        )


def _import_vl_stack() -> tuple[Any, Any, Any]:
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover
        raise RerankerError(
            "Missing dependency 'transformers'/'qwen-vl-utils'. Install backend requirements first."
        ) from exc
    return Qwen3VLForConditionalGeneration, AutoProcessor, process_vision_info


def _import_pil_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RerankerError(
            "Missing dependency 'pillow'. Install backend requirements first."
        ) from exc
    return Image


def _resolve_case_insensitive(root: Path, parts: list[str]) -> Path | None:
    """Resolve ``root/parts`` tolerating per-segment case mismatches.

    The image index stores ``local_path`` values whose folder casing does not
    always match the filesystem (e.g. mj: ``obj_mj_im001_1`` indexed vs
    ``obj_MJ_IM001_1`` on disk).
    """
    current = root
    for part in parts:
        candidate = current / part
        if candidate.exists():
            current = candidate
            continue
        lowered = part.lower()
        try:
            match = next(
                (child for child in current.iterdir() if child.name.lower() == lowered),
                None,
            )
        except OSError:
            return None
        if match is None:
            return None
        current = match
    return current if current.is_file() else None


class VLRerankerService:
    """Second-stage visual reranker following the official Qwen3-VL-Reranker
    transformers usage (QwenLM/Qwen3-VL-Embedding, src/models/qwen3_vl_reranker.py):
    backbone without LM head + binary linear built from lm_head[yes]-lm_head[no],
    sigmoid score over the last hidden state.

    Query and documents may mix modalities: the query is the uploaded image
    (plus any user text) and each document is the candidate image with its
    caption/title. Candidates whose pixels cannot be resolved degrade to
    text-only documents instead of being dropped.
    """

    _SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query and the "
        "Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    )
    _MAX_LENGTH = 10240
    _IMAGE_FACTOR = 32
    _MIN_PIXELS = 4 * 32 * 32
    _MAX_PIXELS = 1800 * 32 * 32

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.vl_reranker_model_resolved
        self.revision = settings.vl_reranker_model_revision_resolved
        self.instruction = settings.VL_RERANKER_INSTRUCTION.strip()
        self.batch_size = max(int(settings.VL_RERANKER_BATCH_SIZE), 1)
        self.max_image_side = max(int(settings.VL_RERANKER_MAX_IMAGE_SIDE), 64)
        self.prefer_bf16 = bool(settings.RERANKER_PREFER_BF16)
        self.device: str | None = None
        self.dtype: Any | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._score_linear: Any | None = None
        self._process_vision_info: Any | None = None

    def _resolve_runtime(self) -> tuple[str, Any]:
        torch = _import_torch()
        if torch.cuda.is_available():
            bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            if self.prefer_bf16 and bf16_supported:
                return "cuda", torch.bfloat16
            return "cuda", torch.float16
        return "cpu", torch.float32

    def _get_model(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor, self._score_linear

        torch = _import_torch()
        Qwen3VLForConditionalGeneration, AutoProcessor, process_vision_info = _import_vl_stack()
        self.device, self.dtype = self._resolve_runtime()

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.revision:
            model_kwargs["revision"] = self.revision
        if self.device != "cpu":
            model_kwargs["dtype"] = self.dtype
        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "padding_side": "left",
        }
        if self.revision:
            processor_kwargs["revision"] = self.revision

        try:
            lm = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
            lm = lm.to(self.device)
            processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
        except Exception as exc:
            raise RerankerError(
                f"Failed to load VL reranker model '{self.model_id}': {exc}"
            ) from exc

        vocab = processor.tokenizer.get_vocab()
        token_yes = vocab.get("yes")
        token_no = vocab.get("no")
        if token_yes is None or token_no is None:
            raise RerankerError("Could not resolve yes/no token ids for VL reranker scoring.")

        # Cabeca binaria oficial: peso = lm_head[yes] - lm_head[no].
        lm_head_weights = lm.lm_head.weight.data
        score_linear = torch.nn.Linear(lm_head_weights.size(1), 1, bias=False)
        with torch.no_grad():
            score_linear.weight[0] = lm_head_weights[token_yes] - lm_head_weights[token_no]

        model = lm.model
        model.eval()
        score_linear.eval()
        score_linear = score_linear.to(self.device).to(next(model.parameters()).dtype)

        self._model = model
        self._processor = processor
        self._score_linear = score_linear
        self._process_vision_info = process_vision_info
        return self._model, self._processor, self._score_linear

    def _build_pair_messages(
        self,
        *,
        query: dict[str, Any],
        document: dict[str, Any],
    ) -> list[dict[str, Any]]:
        def content_block(payload: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
            content: list[dict[str, Any]] = [{"type": "text", "text": prefix}]
            image = payload.get("image")
            text = str(payload.get("text") or "").strip()
            if image is None and not text:
                content.append({"type": "text", "text": "NULL"})
                return content
            if image is not None:
                content.append(
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": self._MIN_PIXELS,
                        "max_pixels": self._MAX_PIXELS,
                    }
                )
            if text:
                content.append({"type": "text", "text": text})
            return content

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": "<Instruct>: " + (self.instruction or "")}
        ]
        user_content.extend(content_block(query, "<Query>:"))
        user_content.extend(content_block(document, "\n<Document>:"))
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._SYSTEM_PROMPT}],
            },
            {"role": "user", "content": user_content},
        ]

    def _score_pairs(self, pairs: list[list[dict[str, Any]]]) -> list[float]:
        torch = _import_torch()
        model, processor, score_linear = self._get_model()
        process_vision_info = self._process_vision_info
        if process_vision_info is None:
            raise RerankerError("VL reranker vision preprocessing is not initialized.")

        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            text = processor.apply_chat_template(
                batch, tokenize=False, add_generation_prompt=True
            )
            images, videos, video_kwargs = process_vision_info(
                batch,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            inputs = processor(
                text=text,
                images=images,
                videos=None,
                truncation=False,
                padding=False,
                do_resize=False,
                **{k: v for k, v in (video_kwargs or {}).items() if k != "fps"},
            )

            # Truncagem de emergencia (nao dispara com os limites de pixels
            # configurados): corta o excesso do fim do prompt preservando os
            # ultimos 5 tokens (generation prompt), em posicoes alinhadas para
            # input_ids e mm_token_type_ids.
            raw_ids = [list(ids) for ids in inputs["input_ids"]]
            raw_types_value = inputs.get("mm_token_type_ids")
            raw_types = (
                [list(types) for types in raw_types_value]
                if raw_types_value is not None
                else None
            )
            for index, ids in enumerate(raw_ids):
                if len(ids) > self._MAX_LENGTH:
                    keep_head = self._MAX_LENGTH - 5
                    raw_ids[index] = [*ids[:keep_head], *ids[-5:]]
                    if raw_types is not None:
                        types = raw_types[index]
                        raw_types[index] = [*types[:keep_head], *types[-5:]]

            # Left padding manual (padding_side=left), mantendo
            # mm_token_type_ids sincronizado — exigido pelo M-RoPE do Qwen3-VL.
            pad_id = processor.tokenizer.pad_token_id or 0
            max_len = max(len(ids) for ids in raw_ids)
            input_rows: list[list[int]] = []
            mask_rows: list[list[int]] = []
            type_rows: list[list[int]] = []
            for index, ids in enumerate(raw_ids):
                pad = max_len - len(ids)
                input_rows.append([pad_id] * pad + ids)
                mask_rows.append([0] * pad + [1] * len(ids))
                if raw_types is not None:
                    type_rows.append([0] * pad + raw_types[index])

            inputs["input_ids"] = torch.tensor(input_rows, dtype=torch.long)
            inputs["attention_mask"] = torch.tensor(mask_rows, dtype=torch.long)
            if raw_types is not None:
                inputs["mm_token_type_ids"] = torch.tensor(type_rows, dtype=torch.long)
            if hasattr(inputs, "to") and self.device:
                inputs = inputs.to(self.device)

            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state[:, -1]
                batch_scores = torch.sigmoid(score_linear(hidden)).squeeze(-1)
            values = batch_scores.detach().cpu().float().reshape(-1).tolist()
            scores.extend(float(value) for value in values)
        return scores

    def _downscale(self, image: Any) -> Any:
        width, height = image.size
        longest = max(width, height)
        if longest <= self.max_image_side:
            return image
        scale = self.max_image_side / float(longest)
        return image.resize(
            (max(int(width * scale), 1), max(int(height * scale), 1))
        )

    def _load_query_image(self, image_bytes: bytes | None) -> Any | None:
        if not image_bytes:
            return None
        Image = _import_pil_image()
        try:
            import io

            with Image.open(io.BytesIO(image_bytes)) as image:
                return self._downscale(image.convert("RGB").copy())
        except Exception:
            return None

    def _resolve_hit_image(self, hit: dict[str, Any]) -> Any | None:
        root = self.settings.image_asset_root_resolved
        if root is None:
            return None
        local_path = str(hit.get("local_path") or "").strip()
        if not local_path:
            return None

        normalized = local_path.replace("\\", "/").strip("/")
        parts = [part for part in Path(normalized).parts if part not in {"", "."}]
        if not parts or any(part == ".." for part in parts):
            return None
        if parts[0].lower() == root.name.lower():
            parts = parts[1:]
        if not parts:
            return None

        target = root / Path(*parts)
        if not target.is_file():
            target = _resolve_case_insensitive(root, parts)
        if target is None or not target.is_file():
            return None

        Image = _import_pil_image()
        try:
            with Image.open(target) as image:
                return self._downscale(image.convert("RGB").copy())
        except Exception:
            return None

    def _build_document(self, hit: dict[str, Any]) -> dict[str, Any]:
        text_bits = [
            str(hit.get("artifact_title") or "").strip(),
            str(hit.get("caption") or "").strip(),
        ]
        text = ". ".join(bit for bit in text_bits if bit)
        document: dict[str, Any] = {}
        image = self._resolve_hit_image(hit)
        if image is not None:
            document["image"] = image
        if text:
            document["text"] = text
        if not document:
            document["text"] = "no-document-content"
        return document

    def _rerank_sync(
        self,
        *,
        query_text: str,
        query_image_bytes: bytes | None,
        image_hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not image_hits:
            return []

        query: dict[str, Any] = {}
        query_image = self._load_query_image(query_image_bytes)
        if query_image is not None:
            query["image"] = query_image
        cleaned_query_text = (query_text or "").strip()
        if cleaned_query_text:
            query["text"] = cleaned_query_text
        if not query:
            return [dict(hit) for hit in image_hits[:top_k]]

        documents = [self._build_document(hit) for hit in image_hits]
        pairs = [
            self._build_pair_messages(query=query, document=document)
            for document in documents
        ]

        try:
            values = self._score_pairs(pairs)
        except RerankerError:
            raise
        except Exception as exc:
            raise RerankerError(f"Failed to score query-image pairs: {exc}") from exc

        if len(values) != len(image_hits):
            raise RerankerError(
                f"Unexpected VL reranker output size. expected={len(image_hits)} got={len(values)}"
            )

        scored_hits: list[dict[str, Any]] = []
        for index, hit in enumerate(image_hits):
            scored = dict(hit)
            scored["vl_rerank_score"] = float(values[index])
            scored["retrieval_rank"] = index + 1
            scored_hits.append(scored)

        scored_hits.sort(key=lambda item: item.get("vl_rerank_score", float("-inf")), reverse=True)
        return scored_hits[:top_k]

    async def rerank_image_hits(
        self,
        *,
        query_text: str,
        query_image_bytes: bytes | None,
        image_hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        effective_top_k = max(int(top_k), 1)
        if not self.settings.CHAT_ENABLE_VL_RERANKING:
            return [dict(hit) for hit in image_hits[:effective_top_k]]
        return await asyncio.to_thread(
            self._rerank_sync,
            query_text=query_text,
            query_image_bytes=query_image_bytes,
            image_hits=image_hits,
            top_k=effective_top_k,
        )

    async def warmup(self) -> None:
        if not self.settings.CHAT_ENABLE_VL_RERANKING:
            return
        await asyncio.to_thread(self._get_model)


@lru_cache(maxsize=1)
def get_reranker_service() -> PairwiseRerankerService:
    return PairwiseRerankerService(get_settings())


@lru_cache(maxsize=1)
def get_vl_reranker_service() -> VLRerankerService:
    return VLRerankerService(get_settings())
