"""Local configuration for the OpenAI-backed entity-resolution oracle.

Prefer setting `OPENAI_API_KEY` in `llm_config_local.py` or exporting it in the
shell environment so local secrets stay out of version control. The environment
variable takes precedence when present.
"""

from __future__ import annotations

import os


OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-5-mini"
PROMPT_MODE = "zero-shot"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_TIMEOUT_SECONDS = 120
OPENAI_MAX_RETRIES = 3
OPENAI_MAX_OUTPUT_TOKENS = 400
OPENAI_REASONING_EFFORT = "low"
OPENAI_EXAMPLES_PER_ENTITY = 3

try:
    from llm_config_local import *  # noqa: F403
except ImportError:
    pass


def get_openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY).strip()


def resolve_model(override: str | None = None) -> str:
    return (override or OPENAI_MODEL).strip()


def resolve_prompt_mode(override: str | None = None) -> str:
    return (override or PROMPT_MODE).strip()
