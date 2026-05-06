from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI

from app.core.config import Settings, get_settings
from app.schemas.chat import ResponseFormatObject


class LLMServiceError(Exception):
    """Raised when the LLM provider fails."""


@dataclass
class LLMResponse:
    text: str
    model: str
    response_format: ResponseFormatObject
    parsed_json: dict[str, Any] | list[Any] | None = None


class LLMService:
    """LLM service for dev calls using OpenAI-compatible chat completions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=self.settings.LLM_API_KEY or "no-key",
            base_url=self.settings.llm_openai_base_url_resolved,
            timeout=self.settings.LLM_TIMEOUT_SECONDS,
            default_headers=self.settings.llm_auth_header or None,
        )

    async def generate(
        self,
        *,
        message: str,
        response_format: ResponseFormatObject,
        system_prompt: str | None = None,
        model_override: str | None = None,
    ) -> LLMResponse:
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
        # if self.settings.LLM_MAX_TOKENS > 0:
        #     payload["max_tokens"] = self.settings.LLM_MAX_TOKENS

        try:
            # Prefer parse-style call when available on the configured SDK/client.
            completions_api = self.client.chat.completions
            if hasattr(completions_api, "parse"):
                completion = await completions_api.parse(**payload)
            else:
                completion = await completions_api.create(**payload)
        except (APIError, APITimeoutError) as exc:
            raise LLMServiceError(f"LLM request failed: {exc}") from exc
        except Exception as exc:
            raise LLMServiceError(f"LLM request failed: {exc}") from exc

        content = _extract_chat_content(completion)
        if not isinstance(content, str) or not content.strip():
            raise LLMServiceError("LLM returned empty content.")

        parsed_json: dict[str, Any] | list[Any] | None = None
        if response_format.type == "json_object":
            parsed_json = _parse_json_output(content)

        return LLMResponse(
            text=content.strip(),
            model=model,
            response_format=response_format,
            parsed_json=parsed_json,
        )


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
    import json

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
