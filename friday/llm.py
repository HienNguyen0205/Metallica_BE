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

#: Measured against the free tier, not taken from docs — every assumption here
#: has been wrong at least once:
#:
#:   gemini-2.5-flash        404, "no longer available to new users"
#:   gemini-3.7-flash        hangs past 45s
#:   gemini-flash-latest     hangs past 45s
#:   gemini-3-flash-preview  InternalServerError
#:   gemini-3.6-flash        works, but only 20 requests PER DAY on free tier
#:   gemini-3.5-flash        works, 15.5s to first tool call
#:   gemini-3.5-flash-lite   works, 0.8s tool call / 1.2s structured output
#:
#: The daily quota is per model, so the choice is a quota decision as much as a
#: speed one: at 20/day and 2-3 calls per query, gemini-3.6-flash allowed about
#: six questions a day. Pinned rather than using the `-latest` alias because an
#: alias can move to a model with a tiny quota without warning, which is
#: exactly how 3.6-flash behaves.
#:
#: When this is retired, the 429/404 body names the quota and the replacement;
#: `models.list()` shows what is currently available.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

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
