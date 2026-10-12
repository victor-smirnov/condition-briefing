"""Thin OpenAI wrapper that returns parsed pydantic objects and records usage for provenance."""

import hashlib
import threading
import time
from functools import cache

from openai import OpenAI
from pydantic import BaseModel

from . import tracing


@cache
def client() -> OpenAI:
    tracing.setup()  # instrument the OpenAI SDK before first use
    return OpenAI()


def prompt_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class Usage:
    """Accumulates LLM usage for one pipeline step. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.models: set[str] = set()
        self.prompts: dict[str, str] = {}  # prompt id -> sha of the system prompt

    def add(self, model: str, prompt_id: str, system: str, usage) -> None:
        with self._lock:
            self.calls += 1
            self.models.add(model)
            self.prompts[prompt_id] = prompt_sha(system)
            if usage:
                self.input_tokens += usage.prompt_tokens
                self.output_tokens += usage.completion_tokens

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "models": sorted(self.models),
            "prompts": self.prompts,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def structured[T: BaseModel](
    *,
    model: str,
    system: str,
    user: str,
    schema: type[T],
    usage: Usage,
    prompt_id: str,
    effort: str = "low",
    retries: int = 2,
) -> T:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = client().chat.completions.parse(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format=schema,
                reasoning_effort=effort,
            )
            usage.add(model, prompt_id, system, r.usage)
            parsed = r.choices[0].message.parsed
            if parsed is None:
                raise RuntimeError(f"model refused or returned nothing: {r.choices[0].message.refusal}")
            return parsed
        except Exception as e:  # network errors, rate limits, parse failures
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call {prompt_id} failed: {last_err}")
