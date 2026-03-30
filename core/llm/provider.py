from __future__ import annotations
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional
import requests

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def generate(self, messages: list[dict], temperature: float = 0.3,
                 max_tokens: int = 2000, system: str = "") -> str:
        """Generate a response from messages."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...

    def chat(self, user_message: str, system: str = "", history: list[dict] = None) -> str:
        """Simple chat interface."""
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return self.generate(messages, system=system)


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider for accessing cloud LLMs.

    Uses the OpenRouter-compatible OpenAI API format.

    Args:
        api_key:     OpenRouter API key (defaults to OPENROUTER_API_KEY env var).
        model:       Model identifier on OpenRouter.
        temperature: Default sampling temperature.
        max_tokens:  Default max generation tokens.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "qwen/qwen-2.5-7b-instruct",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 120,
    ):
        import os
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = "https://openrouter.ai/api/v1"

    def generate(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
        system: str = "",
    ) -> str:
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        if system:
            messages = [{"role": "system", "content": system}] + messages

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        r = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def generate_json(self, messages, schema=None, temperature=0.1):
        response = self.generate(messages, temperature=temperature)
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider.

    Connects to a running Ollama instance and provides:
      - Chat-based generation with configurable temperature/max_tokens
      - JSON-mode generation with automatic retry and markdown stripping
      - Token counting via tiktoken (falls back to char-based heuristic)
      - Model pulling utility

    Example::

        llm = OllamaProvider(model="qwen2.5:4b-instruct")
        reply = llm.chat("What is the capital of France?")
    """

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.timeout = timeout
        self._check_connection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_connection(self) -> None:
        """Verify Ollama is running and warn if the model is absent."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if self.model not in models:
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s",
                    self.model,
                    models,
                )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
        system: str = "",
    ) -> str:
        """Generate a response via the Ollama ``/api/chat`` endpoint.

        Args:
            messages:    List of ``{"role": ..., "content": ...}`` dicts.
            temperature: Sampling temperature (defaults to instance default).
            max_tokens:  Maximum tokens to generate (defaults to instance default).
            system:      Optional system prompt injected into the request.

        Returns:
            The model's text response as a plain string.

        Raises:
            TimeoutError:  When Ollama does not respond within ``self.timeout`` seconds.
            RuntimeError:  On any other HTTP/request error.
        """
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": tokens,
            },
        }
        if system:
            payload["system"] = system

        start = time.time()
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            content: str = data["message"]["content"]
            elapsed = time.time() - start

            prompt_tokens = data.get("prompt_eval_count", 0)
            response_tokens = data.get("eval_count", 0)
            logger.debug(
                "LLM: %d+%d tokens, %.1fs", prompt_tokens, response_tokens, elapsed
            )

            return content

        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama timed out after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}") from e

    def generate_json(
        self,
        messages: list[dict],
        schema: dict = None,
        temperature: float = 0.1,
    ) -> "dict | list":
        """Generate a response and parse it as JSON.

        Strips Markdown code fences if present and retries up to 3 times on
        ``json.JSONDecodeError``, asking the model to correct itself each time.

        Args:
            messages:    Conversation history / prompt.
            schema:      Optional JSON schema (currently informational only — will be
                         embedded in a future structured-output call).
            temperature: Low temperature recommended for deterministic JSON output.

        Returns:
            Parsed Python ``dict`` or ``list``.  Returns ``{}`` / ``[]`` on
            persistent failure (logged at ERROR level).
        """
        current_messages = list(messages)

        for attempt in range(3):
            response = self.generate(current_messages, temperature=temperature)
            try:
                text = response.strip()
                # Strip Markdown code fences
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                return json.loads(text)
            except json.JSONDecodeError:
                if attempt < 2:
                    current_messages = current_messages + [
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": (
                                "That was not valid JSON. "
                                "Please return ONLY valid JSON, no markdown."
                            ),
                        },
                    ]
                    logger.warning(
                        "JSON parse failed (attempt %d), retrying...", attempt + 1
                    )

        logger.error(
            "Failed to parse JSON after 3 attempts. Response: %s", response[:200]
        )
        return [] if "[" in response else {}

    def count_tokens(self, text: str) -> int:
        """Count tokens in *text*.

        Uses ``tiktoken`` (cl100k_base) when available; falls back to the
        ``len(text) // 4`` heuristic otherwise.
        """
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return len(text) // 4

    @classmethod
    def from_env(cls) -> "LLMProvider":
        """Factory that returns an OpenRouterProvider when OPENROUTER_API_KEY is set,
        otherwise returns an OllamaProvider.

        Returns:
            An LLMProvider instance configured from environment variables.
        """
        import os
        if os.environ.get("OPENROUTER_API_KEY"):
            model = os.environ.get("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")
            return OpenRouterProvider(
                api_key=os.environ["OPENROUTER_API_KEY"],
                model=model,
            )
        model = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return cls(model=model, base_url=base_url)

    def pull_model(self) -> None:
        """Pull the configured model from the Ollama registry.

        Streams progress to stdout so the caller can observe download status.
        """
        logger.info("Pulling model %s...", self.model)
        r = requests.post(
            f"{self.base_url}/api/pull",
            json={"name": self.model},
            stream=True,
            timeout=300,
        )
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if "status" in data:
                    print(f"\r{data['status']}", end="", flush=True)
        print()
