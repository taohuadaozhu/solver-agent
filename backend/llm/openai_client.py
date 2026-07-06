import logging
import os
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import httpx
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required. Set it in your environment or .env file.")

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1")

REQUEST_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "30.0"))
MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
            max_retries=0,  # we handle retries ourselves via tenacity
        )
    return _client


def _retry_predicate(exception: Exception) -> bool:
    if isinstance(exception, httpx.TimeoutException):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
    return False


_retry_decorator = retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    before_sleep=lambda retry_state: logger.warning(
        "OpenAI API retry %d/%d after %.1fs — %s",
        retry_state.attempt_number,
        MAX_RETRIES,
        retry_state.idle_for,
        retry_state.outcome.exception() if retry_state.outcome else "unknown",
    ),
)


@_retry_decorator
async def embeddings(
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    **kwargs,
) -> List[List[float]]:
    logger.debug("Embedding %d texts with model=%s", len(texts), model)
    client = _get_client()
    response = await client.embeddings.create(model=model, input=texts, **kwargs)
    return [item.embedding for item in response.data]


@_retry_decorator
async def chat_completion(
    prompt: str,
    model: str = CHAT_MODEL,
    system: str = "You are a helpful algorithm recommendation assistant.",
    **kwargs,
) -> str:
    logger.debug("Chat completion with model=%s", model)
    client = _get_client()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        **kwargs,
    )
    return response.choices[0].message.content
