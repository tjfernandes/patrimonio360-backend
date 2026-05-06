from __future__ import annotations

import asyncio
from functools import lru_cache
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

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if self.device != "cpu":
            model_kwargs["torch_dtype"] = self.dtype
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


@lru_cache(maxsize=1)
def get_reranker_service() -> PairwiseRerankerService:
    return PairwiseRerankerService(get_settings())
