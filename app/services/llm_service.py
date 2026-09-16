import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Awaitable, Callable

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI

from app.core.config import Settings, get_settings
from app.core.logging import log_event
from app.schemas.chat import ResponseFormatObject

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Raised when the LLM provider fails."""


@dataclass
class LLMResponse:
    text: str
    model: str
    response_format: ResponseFormatObject
    parsed_json: dict[str, Any] | list[Any] | None = None
    # Quick win LLM-04: motivo de paragem e contagem de tokens quando o
    # servidor os devolve ("stop" | "length" | ...). None = não reportado.
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    # "primary" (LLM_BASE_URL) ou "fallback" (LLM_FALLBACK_BASE_URL).
    endpoint: str = "primary"


class LLMService:
    """LLM service for dev calls using OpenAI-compatible chat completions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        timeout = httpx.Timeout(
            float(self.settings.LLM_TIMEOUT_SECONDS),
            connect=float(getattr(self.settings, "LLM_CONNECT_TIMEOUT_SECONDS", 5.0) or 5.0),
        )
        self.client = AsyncOpenAI(
            api_key=self.settings.LLM_API_KEY or "no-key",
            base_url=self.settings.llm_openai_base_url_resolved,
            timeout=timeout,
            default_headers=self.settings.llm_auth_header or None,
        )
        # Reserva (ex.: API remota quando o primário é o vLLM local). Só é
        # usada quando o primário não aceita a ligação — nunca por erros de
        # conteúdo (JSON inválido, 4xx), que se repetiriam no fallback.
        self.fallback_client: AsyncOpenAI | None = None
        self.fallback_model: str | None = (getattr(self.settings, "LLM_FALLBACK_MODEL", "") or "").strip() or None
        fallback_base = getattr(self.settings, "llm_fallback_openai_base_url_resolved", None)
        if fallback_base:
            fallback_key = getattr(self.settings, "LLM_FALLBACK_API_KEY", None) or self.settings.LLM_API_KEY
            self.fallback_client = AsyncOpenAI(
                api_key=fallback_key or "no-key",
                base_url=fallback_base,
                timeout=float(self.settings.LLM_TIMEOUT_SECONDS),
                default_headers={"Authorization": f"Bearer {fallback_key}"} if fallback_key else None,
            )

    def _fallback_payload(self, payload: dict[str, Any], *, model_override: str | None) -> dict[str, Any]:
        fallback_payload = dict(payload)
        if self.fallback_model and not model_override:
            fallback_payload["model"] = self.fallback_model
        return fallback_payload

    @staticmethod
    async def _complete(client: Any, payload: dict[str, Any]) -> Any:
        # `create`, não `parse`: o `parse` do SDK lança LengthFinishReasonError
        # quando a resposta atinge max_tokens, e essa mensagem («length limit
        # was reached … CompletionUsage … prompt_tokens») era confundida com
        # contexto excessivo, disparando até N regenerações completas (medido
        # em 16-09: 27 s numa pesquisa imagem+texto). O JSON é validado por
        # _parse_json_output, pelo que o `parse` não acrescentava nada.
        return await client.chat.completions.create(**payload)

    def _resolve_max_tokens(self, format_type: str) -> int | None:
        """Orçamento de saída por tipo de chamada (quick win LLM-04).

        Ordem: LLM_MAX_TOKENS_JSON / LLM_MAX_TOKENS_TEXT (por formato) e, na
        sua ausência, LLM_MAX_TOKENS global. 0/None mantém o comportamento
        histórico: nenhum limite enviado ao servidor.
        """
        specific = (
            self.settings.LLM_MAX_TOKENS_JSON
            if format_type == "json_object"
            else self.settings.LLM_MAX_TOKENS_TEXT
        )
        if specific is not None and int(specific) > 0:
            return int(specific)
        if int(self.settings.LLM_MAX_TOKENS or 0) > 0:
            return int(self.settings.LLM_MAX_TOKENS)
        return None

    def _build_payload(
        self,
        *,
        message: str,
        response_format: ResponseFormatObject,
        system_prompt: str | None,
        model_override: str | None,
    ) -> tuple[dict[str, Any], str, str, str]:
        provider = self.settings.LLM_PROVIDER.strip().lower()
        if provider != "openai_compatible":
            raise LLMServiceError(
                f"Unsupported LLM_PROVIDER '{self.settings.LLM_PROVIDER}'. Use 'openai_compatible'."
            )

        model = model_override or self.settings.llm_model_resolved
        temperature = (
            self.settings.LLM_TEMPERATURE_JSON
            if response_format.type == "json_object"
            else self.settings.LLM_TEMPERATURE_TEXT
        )

        user_content = (message or "").strip()
        if not user_content:
            raise LLMServiceError("LLM prompt is empty.")
        system_content = (system_prompt or "").strip()

        messages: list[dict[str, str]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {"type": response_format.type},
            "temperature": temperature,
        }
        max_tokens = self._resolve_max_tokens(response_format.type)
        if max_tokens:
            payload["max_tokens"] = max_tokens
        return payload, provider, model, user_content

    async def generate(
        self,
        *,
        message: str,
        response_format: ResponseFormatObject,
        system_prompt: str | None = None,
        model_override: str | None = None,
    ) -> LLMResponse:
        payload, provider, model, user_content = self._build_payload(
            message=message,
            response_format=response_format,
            system_prompt=system_prompt,
            model_override=model_override,
        )
        system_content = (system_prompt or "").strip()

        started_at = time.perf_counter()
        log_event(
            logger,
            logging.INFO,
            "llm.generate.start",
            provider=provider,
            model=model,
            response_format=response_format.type,
            prompt_chars=len(user_content),
            has_system_prompt=bool(system_content),
            max_tokens=payload.get("max_tokens"),
            timeout_seconds=self.settings.LLM_TIMEOUT_SECONDS,
        )
        endpoint = "primary"
        try:
            try:
                completion = await self._complete(self.client, payload)
            except APIConnectionError as exc:
                if self.fallback_client is None:
                    raise
                log_event(
                    logger,
                    logging.WARNING,
                    "llm.generate.fallback",
                    model=model,
                    fallback_model=self._fallback_payload(payload, model_override=model_override)["model"],
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
                    error=exc,
                )
                endpoint = "fallback"
                completion = await self._complete(
                    self.fallback_client,
                    self._fallback_payload(payload, model_override=model_override),
                )
        except (APIError, APITimeoutError) as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "llm.generate.error",
                model=model,
                duration_ms=round(duration_ms, 1),
                error=exc,
            )
            raise LLMServiceError(f"LLM request failed: {exc}") from exc
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "llm.generate.error",
                model=model,
                duration_ms=round(duration_ms, 1),
                error=exc,
            )
            raise LLMServiceError(f"LLM request failed: {exc}") from exc

        content = _extract_chat_content(completion)
        if not isinstance(content, str) or not content.strip():
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "llm.generate.empty_response",
                model=model,
                duration_ms=round(duration_ms, 1),
            )
            raise LLMServiceError("LLM returned empty content.")
        parsed_json: dict[str, Any] | list[Any] | None = None
        if response_format.type == "json_object":
            try:
                parsed_json = _parse_json_output(content)
            except LLMServiceError as exc:
                duration_ms = (time.perf_counter() - started_at) * 1000
                log_event(
                    logger,
                    logging.ERROR,
                    "llm.generate.parse_error",
                    model=model,
                    duration_ms=round(duration_ms, 1),
                    error=exc,
                )
                raise
        duration_ms = (time.perf_counter() - started_at) * 1000
        finish_reason = _extract_finish_reason(completion)
        usage = _extract_usage(completion)
        log_event(
            logger,
            logging.INFO,
            "llm.generate.finish",
            model=model,
            response_format=response_format.type,
            response_chars=len(content.strip()),
            finish_reason=finish_reason,
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            completion_tokens=(usage or {}).get("completion_tokens"),
            endpoint=endpoint,
            duration_ms=round(duration_ms, 1),
        )

        return LLMResponse(
            text=content.strip(),
            model=model,
            response_format=response_format,
            parsed_json=parsed_json,
            finish_reason=finish_reason,
            usage=usage,
            endpoint=endpoint,
        )

    async def generate_stream(
        self,
        *,
        message: str,
        response_format: ResponseFormatObject,
        system_prompt: str | None = None,
        model_override: str | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Geração com `stream=True` (quick win LLM-03).

        Cada delta de texto é entregue a `on_delta` assim que chega; no fim
        devolve o mesmo LLMResponse que `generate` (texto completo, JSON
        parseado se pedido, finish_reason/usage quando o servidor os envia).
        Só é usado quando CHAT_STREAM_FINAL_ANSWER está ligado; a chamada
        não-stream continua intacta.
        """
        payload, provider, model, user_content = self._build_payload(
            message=message,
            response_format=response_format,
            system_prompt=system_prompt,
            model_override=model_override,
        )
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        started_at = time.perf_counter()
        log_event(
            logger,
            logging.INFO,
            "llm.generate.start",
            provider=provider,
            model=model,
            response_format=response_format.type,
            prompt_chars=len(user_content),
            has_system_prompt=bool((system_prompt or "").strip()),
            max_tokens=payload.get("max_tokens"),
            stream=True,
            timeout_seconds=self.settings.LLM_TIMEOUT_SECONDS,
        )
        parts: list[str] = []
        finish_reason: str | None = None
        usage: dict[str, int] | None = None
        first_delta_ms: float | None = None
        endpoint = "primary"
        try:
            try:
                # Um erro de ligação acontece aqui, antes de qualquer delta —
                # o fallback nunca mistura texto de dois servidores.
                stream = await self.client.chat.completions.create(**payload)
            except APIConnectionError as exc:
                if self.fallback_client is None:
                    raise
                log_event(
                    logger,
                    logging.WARNING,
                    "llm.generate.fallback",
                    model=model,
                    stream=True,
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
                    error=exc,
                )
                endpoint = "fallback"
                stream = await self.fallback_client.chat.completions.create(
                    **self._fallback_payload(payload, model_override=model_override)
                )
            async for chunk in stream:
                chunk_usage = _extract_usage(chunk)
                if chunk_usage:
                    usage = chunk_usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(getattr(choice, "delta", None), "content", None)
                if isinstance(delta, str) and delta:
                    if first_delta_ms is None:
                        first_delta_ms = (time.perf_counter() - started_at) * 1000
                    parts.append(delta)
                    if on_delta is not None:
                        await on_delta(delta)
                reason = getattr(choice, "finish_reason", None)
                if isinstance(reason, str) and reason:
                    finish_reason = reason
        except (APIError, APITimeoutError) as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "llm.generate.error",
                model=model,
                duration_ms=round(duration_ms, 1),
                stream=True,
                error=exc,
            )
            raise LLMServiceError(f"LLM request failed: {exc}") from exc
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "llm.generate.error",
                model=model,
                duration_ms=round(duration_ms, 1),
                stream=True,
                error=exc,
            )
            raise LLMServiceError(f"LLM request failed: {exc}") from exc

        content = "".join(parts)
        if not content.strip():
            log_event(
                logger,
                logging.ERROR,
                "llm.generate.empty_response",
                model=model,
                stream=True,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
            )
            raise LLMServiceError("LLM returned empty content.")
        parsed_json: dict[str, Any] | list[Any] | None = None
        if response_format.type == "json_object":
            parsed_json = _parse_json_output(content)
        duration_ms = (time.perf_counter() - started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "llm.generate.finish",
            model=model,
            response_format=response_format.type,
            response_chars=len(content.strip()),
            finish_reason=finish_reason,
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            completion_tokens=(usage or {}).get("completion_tokens"),
            first_delta_ms=None if first_delta_ms is None else round(first_delta_ms, 1),
            stream=True,
            endpoint=endpoint,
            duration_ms=round(duration_ms, 1),
        )
        return LLMResponse(
            text=content.strip(),
            model=model,
            response_format=response_format,
            parsed_json=parsed_json,
            finish_reason=finish_reason,
            usage=usage,
            endpoint=endpoint,
        )


def _extract_finish_reason(completion: Any) -> str | None:
    choices = getattr(completion, "choices", None)
    if not choices:
        return None
    reason = getattr(choices[0], "finish_reason", None)
    return reason if isinstance(reason, str) and reason else None


def _extract_usage(completion: Any) -> dict[str, int] | None:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            result[key] = value
    return result or None


def _extract_chat_content(completion: Any) -> str | None:
    choices = getattr(completion, "choices", None)
    if not choices:
        return None

    message = getattr(choices[0], "message", None)
    if message is None:
        return None

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)

    return None


def _parse_json_output(raw: str) -> dict[str, Any] | list[Any]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = _strip_fenced_block(candidate)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMServiceError(f"Model did not return valid JSON: {exc}") from exc

    if not isinstance(parsed, (dict, list)):
        raise LLMServiceError("JSON response must be an object or an array.")
    return parsed


def _strip_fenced_block(value: str) -> str:
    lines = value.splitlines()
    if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return value


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService(get_settings())
