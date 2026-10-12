"""LLM call tracing to Arize Phoenix via OpenInference.

Enabled only when PHOENIX_COLLECTOR_ENDPOINT is set (e.g. http://phoenix:6006/v1/traces).
Every OpenAI request then becomes a span with prompt, response, tool calls, tokens and latency.
"""

import os
from functools import cache


@cache
def setup() -> bool:
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return False
    from phoenix.otel import register

    # Pass endpoint and protocol explicitly: from the env var alone, arize-phoenix-otel 0.17
    # switches to port 4317 while keeping the HTTP transport, and the export fails.
    register(project_name=os.getenv("PHOENIX_PROJECT_NAME", "briefing"), endpoint=endpoint,
             protocol="http/protobuf", auto_instrument=True, batch=True)
    return True
