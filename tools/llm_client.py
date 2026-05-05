import os
from openai import OpenAI
from observability.logger import get_logger, log_event

_client: OpenAI | None = None
log = get_logger(__name__)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def generate_response(system_prompt: str, user_prompt: str):
    log_event(log, "llm_call_started", model="gpt-4o-mini")
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",  # fast + cheap for demo
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    content = response.choices[0].message.content
    log_event(log, "llm_call_completed", model="gpt-4o-mini", response_chars=len(content or ""))
    return content