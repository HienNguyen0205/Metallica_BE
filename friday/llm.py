"""§8 Model gateway — one place that knows which provider we talk to.

Everything goes through an OpenAI-compatible endpoint, so the provider is a URL
and a model name rather than a code change. Gemini, Groq, Cerebras, OpenRouter
and a local Ollama all speak this dialect; swapping between them is env only.

That is the whole point of §8 ("để không khóa provider"). It is deliberately a
gateway and not a router: there is no latency/cost scoring here, because there
is nothing yet to score.
"""

import os

from openai import AsyncOpenAI

#: Google AI Studio's OpenAI-compatible surface. Free tier, no card.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

#: Measured, not assumed. `gemini-2.5-flash` 404s ("no longer available to new
#: users"); `gemini-3.7-flash` and the `gemini-flash-latest` alias both hang
#: past 45s on the free tier; this one answers in ~3s. When it is eventually
#: retired the 404 says which model replaced it — see `models.list()`.
DEFAULT_MODEL = "gemini-3.6-flash"

#: An interactive HUD cannot sit on the SDK's 10-minute default. A model that
#: has not answered by now is not going to feel like a response anyway.
REQUEST_TIMEOUT_S = 60.0


def base_url() -> str:
    return os.getenv("FRIDAY_LLM_BASE_URL", GEMINI_BASE_URL)


def model() -> str:
    return os.getenv("FRIDAY_LLM_MODEL", DEFAULT_MODEL)


def api_key() -> str | None:
    # GEMINI_API_KEY is what AI Studio calls it; the generic name wins so
    # pointing at Groq or a local model does not need a Gemini-shaped variable.
    return os.getenv("FRIDAY_LLM_API_KEY") or os.getenv("GEMINI_API_KEY")


def configured() -> bool:
    return bool(api_key())


def client() -> AsyncOpenAI:
    """A client for the configured provider.

    Local runtimes (Ollama, llama.cpp) ignore the key but the SDK requires a
    non-empty one, hence the placeholder.
    """
    return AsyncOpenAI(
        base_url=base_url(),
        api_key=api_key() or "not-needed",
        timeout=REQUEST_TIMEOUT_S,
    )
