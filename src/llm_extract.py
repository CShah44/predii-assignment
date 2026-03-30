import json
from typing import Sequence
from urllib import error, request


def build_extraction_prompt(query: str, contexts: Sequence[str]) -> str:
    """Builds the prompt string to explicitly guide the LLM for structured JSON extraction."""
    joined_context = "\n\n".join(contexts)
    return (
        "You are an expert automotive assistant.\n"
        "Your task is to accurately extract the answer to the user's query using ONLY the provided context.\n"
        "Return the extracted information strictly as a JSON array of objects. Do not include any explanation or markdown formatting.\n\n"
        "Each object in the array represents a piece of the answer. Use the following schema:\n"
        "{\n"
        '  "action_or_component": "The main focus of the step, part, or action (e.g., Replace bolt, Engine oil)",\n'
        '  "description": "Details about what to do, what it is, or the specification",\n'
        '  "value": "Optional numeric value if applicable",\n'
        '  "unit": "Optional unit of measurement if applicable"\n'
        "}\n\n"
        "If the context does not contain the answer, return an empty array: []\n\n"
        f"User query:\n{query}\n\n"
        f"Context:\n{joined_context}\n"
    )


def call_ollama_generate(
    prompt: str,
    model: str,
    base_url: str,
    timeout_seconds: int,
) -> str:
    """Calls a local Ollama API endpoint to generate JSON according to a strict schema."""
    endpoint = base_url.rstrip("/") + "/api/generate"
    
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "action_or_component": {"type": "string"},
                "description": {"type": "string"},
                "value": {"type": ["string", "null"]},
                "unit": {"type": ["string", "null"]}
            },
            "required": ["action_or_component", "description"]
        }
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "format": schema,
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama request timed out after {timeout_seconds}s. "
            "Try increasing --ollama-timeout."
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to call Ollama endpoint {endpoint}: {exc}") from exc

    decoded = json.loads(body)
    return (decoded.get("response") or "").strip()

def extract_structured_specs(
    query: str,
    contexts: Sequence[str],
    model: str,
    base_url: str,
    timeout_seconds: int,
) -> str:
    """Orchestrates building the extraction prompt and calling the Ollama LLM to return strictly structured data."""
    prompt = build_extraction_prompt(query=query, contexts=contexts)
    return call_ollama_generate(
        prompt=prompt,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
