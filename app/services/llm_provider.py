"""Optional LLM provider clients.

These are the concrete, swappable LLM implementations used only when the LLM
fallback is enabled in settings. They implement the abstract interfaces from
:mod:`app.services.classifier` and :mod:`app.services.extractor`.

Security: providers only ever receive the masked/sanitized message, never a raw
sensitive value. API keys are never logged or serialized.

The default configuration keeps the pipeline fully offline: when ``llm_enabled``
is False (or no key/model is configured), :func:`build_llm_components` returns
``(None, None)`` so the deterministic rule paths are always used and no LLM is
required.
"""

from __future__ import annotations

from typing import Final

import httpx

from app.config import Settings
from app.services.classifier import BaseLLMClassifier, MessageClassifierLLM
from app.services.extractor import BaseLLMExtractor, MessageExtractorLLM

_OPENAI_COMPLETIONS_PATH: Final = "/chat/completions"


def build_llm_components(
    settings: Settings,
) -> tuple[MessageClassifierLLM | None, MessageExtractorLLM | None]:
    """Build the classifier/extractor LLM providers from settings.

    Returns ``(None, None)`` (fully offline) when the LLM fallback is disabled,
    or when no API key or model is configured. The provider name determines the
    concrete client; unknown providers fall back to ``None`` rather than crash.
    """
    if not settings.llm_configured:
        return None, None

    if settings.llm_provider != "openai":
        return None, None

    classifier = OpenAIClassifierLLM(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
    )
    extractor = OpenAIExtractorLLM(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
    )
    return classifier, extractor


class _OpenAIBase:
    """Shared OpenAI-compatible chat completions invocation."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    def _invoke(self, *, prompt: str) -> str:
        """POST the prompt to the provider and return the text content."""
        url = self._base_url + _OPENAI_COMPLETIONS_PATH
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"])


class OpenAIClassifierLLM(_OpenAIBase, BaseLLMClassifier):
    """OpenAI-compatible classification provider."""


class OpenAIExtractorLLM(_OpenAIBase, BaseLLMExtractor):
    """OpenAI-compatible extraction provider."""
