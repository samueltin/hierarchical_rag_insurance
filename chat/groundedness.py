"""
Groundedness checking via Azure AI Content Safety.

Asks the service whether an answer is supported by the policy text it was given.
Ungrounded sentences are the failure mode that matters here: a fluent answer
citing a real section but stating a figure that is not in it.

Configuration (.env):
    AZURE_CONTENT_SAFETY_ENDPOINT   https://<resource>.cognitiveservices.azure.com/
    AZURE_CONTENT_SAFETY_KEY        resource key
    GROUNDEDNESS_SOURCES            "all" (default) or "best"
    GROUNDEDNESS_ENABLED            "false" to switch the check off

The check is advisory: if it is not configured, errors, or times out, the answer
is still returned. A groundedness service being down must not take chat down.
"""

from dataclasses import dataclass, field
import os

import requests

API_VERSION = os.getenv("CONTENT_SAFETY_API_VERSION", "2024-09-15-preview")
TIMEOUT = int(os.getenv("GROUNDEDNESS_TIMEOUT", "30"))

# The service caps the grounding sources it will accept.
MAX_SOURCE_CHARS = 10_000
MAX_TOTAL_CHARS = 50_000


@dataclass
class GroundednessResult:
    """Outcome of one check. `checked` is False when no verdict was obtained."""

    checked: bool = False
    ungrounded: bool = False
    ungrounded_percentage: float = 0.0
    ungrounded_text: list[str] = field(default_factory=list)
    sources_used: int = 0
    detail: str = ""          # why the check did not run, when it did not

    @property
    def grounded_percentage(self) -> float:
        """0.0 when no check ran — an unchecked answer must not read as 100% grounded."""
        if not self.checked:
            return 0.0
        return round(100.0 * (1.0 - self.ungrounded_percentage), 1)

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "ungrounded": self.ungrounded,
            "ungrounded_percentage": self.ungrounded_percentage,
            "grounded_percentage": self.grounded_percentage,
            "ungrounded_text": self.ungrounded_text,
            "sources_used": self.sources_used,
            "detail": self.detail,
        }


def is_configured() -> bool:
    return bool(
        os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT") and os.getenv("AZURE_CONTENT_SAFETY_KEY")
    )


def is_enabled() -> bool:
    return os.getenv("GROUNDEDNESS_ENABLED", "true").lower() != "false" and is_configured()


def select_sources(source_texts: list[str], mode: str | None = None) -> list[str]:
    """
    Which retrieved sections to check the answer against.

    "all" (the default) checks against every section the model was given, which
    is what groundedness means here — an answer drawing on the second and third
    sections is grounded, not hallucinated. "best" checks against the top
    section only, which is stricter and will flag legitimate answers that
    combined several sections.
    """
    mode = (mode or os.getenv("GROUNDEDNESS_SOURCES", "all")).lower()
    chosen = source_texts[:1] if mode == "best" else source_texts

    trimmed, total = [], 0
    for text in chosen:
        if not text:
            continue
        clipped = text[:MAX_SOURCE_CHARS]
        if total + len(clipped) > MAX_TOTAL_CHARS:
            break
        trimmed.append(clipped)
        total += len(clipped)
    return trimmed


def check(query: str, answer: str, source_texts: list[str],
          mode: str | None = None) -> GroundednessResult:
    """
    Check `answer` against the policy sections it was drawn from.

    Never raises: a failed check returns checked=False with the reason, so the
    caller can show the answer and note that it could not be verified.
    """
    if not is_enabled():
        reason = (
            "GROUNDEDNESS_ENABLED=false"
            if is_configured()
            else "AZURE_CONTENT_SAFETY_ENDPOINT / AZURE_CONTENT_SAFETY_KEY not set"
        )
        return GroundednessResult(detail=reason)

    sources = select_sources(source_texts, mode)
    if not sources or not answer.strip():
        return GroundednessResult(detail="Nothing to check")

    endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT", "").rstrip("/")
    url = f"{endpoint}/contentsafety/text:detectGroundedness?api-version={API_VERSION}"
    payload = {
        "domain": "Generic",
        "task": "QnA",
        "qna": {"query": query},
        "text": answer,
        "groundingSources": sources,
        "reasoning": False,     # reasoning=True additionally requires an Azure OpenAI resource
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Ocp-Apim-Subscription-Key": os.getenv("AZURE_CONTENT_SAFETY_KEY", "")},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        return GroundednessResult(detail=f"{type(exc).__name__}: {exc}")

    if response.status_code >= 400:
        return GroundednessResult(detail=f"HTTP {response.status_code}: {response.text[:300]}")

    data = response.json()
    return GroundednessResult(
        checked=True,
        ungrounded=bool(data.get("ungroundedDetected", False)),
        ungrounded_percentage=float(data.get("ungroundedPercentage", 0.0)),
        ungrounded_text=[d.get("text", "") for d in data.get("ungroundedDetails", []) if d.get("text")],
        sources_used=len(sources),
    )
