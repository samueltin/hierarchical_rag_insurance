"""
Conversation transcripts, stored one JSON blob per conversation.

    chat-history/{user}/{updated_at}-{conversation_id}.json

Listing a user's conversations reads blob *metadata* only — the title and
timestamp live there — so the sidebar costs one listing call rather than one
download per conversation.

Writes use the blob's ETag: the server owns appends, but two browser tabs on the
same conversation would otherwise silently overwrite each other.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
import os
import re
import uuid

from azure.core import MatchConditions
from azure.core.exceptions import ResourceModifiedError, ResourceNotFoundError
from azure.storage.blob import ContentSettings

from storage import (
    JSON_CONTENT_TYPE,
    BlobLocation,
    account_url,
    blob_client,
    container_client,
    download_blob,
    ensure_container,
)

CONTAINER = os.getenv("CHAT_HISTORY_CONTAINER", "chat-history")
MAX_TITLE = 60

# A user id becomes a blob path segment: reject anything that could escape it.
_UNSAFE = re.compile(r"[/\\#?\x00-\x1f]")


class ConversationConflict(RuntimeError):
    """The conversation changed underneath us — reload and retry."""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_user(user: str) -> str:
    candidate = (user or "").strip()
    if not candidate or _UNSAFE.search(candidate) or candidate.startswith(".."):
        raise ValueError(f"Invalid user id: {user!r}")
    return candidate


@dataclass
class SourceRef:
    """
    A citation on an assistant message.

    Deliberately holds no chunk text: a parent can be 4,500 characters, five per
    answer, and the whole transcript is re-read on every turn. Fetch the text
    from `parent_path` if it needs displaying.
    """

    section: str = ""
    document_name: str = ""
    parent_path: str = ""
    score: float = 0.0     # combined score of this parent's matching chunks
    hits: int = 0          # how many child chunks matched


@dataclass
class Message:
    role: str                      # "user" | "assistant"
    content: str
    ts: str = field(default_factory=utcnow)
    sources: list[SourceRef] = field(default_factory=list)
    document_name: str = ""        # scope this turn was answered under, "" = all documents
    search_query: str = ""         # set when a follow-up was condensed before searching
    groundedness: dict = field(default_factory=dict)   # Content Safety verdict, if checked

    @classmethod
    def from_dict(cls, raw: dict) -> "Message":
        return cls(
            role=raw.get("role", ""),
            content=raw.get("content", ""),
            ts=raw.get("ts", ""),
            sources=[SourceRef(**s) for s in raw.get("sources", [])],
            document_name=raw.get("document_name", ""),
            search_query=raw.get("search_query", ""),
            groundedness=raw.get("groundedness", {}) or {},
        )


@dataclass
class Conversation:
    conversation_id: str
    user: str
    title: str = "New chat"
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    messages: list[Message] = field(default_factory=list)

    # Set when loaded from storage, used for the conditional write.
    etag: str | None = None
    blob_name: str | None = None

    def to_json(self) -> str:
        payload = {
            "conversation_id": self.conversation_id,
            "user": self.user,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [asdict(m) for m in self.messages],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: bytes) -> "Conversation":
        data = json.loads(raw.decode("utf-8"))
        return cls(
            conversation_id=data["conversation_id"],
            user=data.get("user", ""),
            title=data.get("title", "New chat"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
        )

    def turns(self, limit: int = 6) -> list[Message]:
        """The most recent messages, for prompting. Older turns are dropped."""
        return self.messages[-limit * 2:] if limit else list(self.messages)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _location(user: str, blob_name: str) -> BlobLocation:
    return BlobLocation(account_url(), CONTAINER, f"{safe_user(user)}/{blob_name}")


def _blob_name(conversation_id: str) -> str:
    return f"{conversation_id}.json"


def create(user: str, title: str = "New chat") -> Conversation:
    conversation = Conversation(
        conversation_id=str(uuid.uuid4()), user=safe_user(user), title=title
    )
    save(conversation, expect_existing=False)
    return conversation


def load(user: str, conversation_id: str) -> Conversation | None:
    location = _location(user, _blob_name(conversation_id))
    try:
        client = blob_client(location)
        stream = client.download_blob()
        conversation = Conversation.from_json(stream.readall())
    except ResourceNotFoundError:
        return None
    conversation.etag = stream.properties.etag
    conversation.blob_name = location.filename
    return conversation


def save(conversation: Conversation, expect_existing: bool = True) -> Conversation:
    """
    Write the transcript back.

    The title and updated_at also go into blob metadata so that listing a user's
    conversations does not need to open any of them.
    """
    conversation.updated_at = utcnow()
    location = _location(conversation.user, _blob_name(conversation.conversation_id))
    ensure_container(location)
    client = blob_client(location)

    kwargs = {
        "data": conversation.to_json().encode("utf-8"),
        "overwrite": True,
        "content_settings": ContentSettings(content_type=JSON_CONTENT_TYPE),
        "metadata": {
            "title": conversation.title[:MAX_TITLE],
            "updated_at": conversation.updated_at,
            "messages": str(len(conversation.messages)),
        },
    }
    if expect_existing and conversation.etag:
        kwargs["etag"] = conversation.etag
        kwargs["match_condition"] = MatchConditions.IfNotModified

    try:
        result = client.upload_blob(**kwargs)
    except ResourceModifiedError as exc:
        raise ConversationConflict(
            f"Conversation {conversation.conversation_id} was updated elsewhere"
        ) from exc
    conversation.etag = result.get("etag")
    return conversation


def list_conversations(user: str) -> list[dict]:
    """Newest first. Reads metadata only — no transcript is downloaded."""
    prefix = f"{safe_user(user)}/"
    container = container_client(_location(user, ""))
    if not container.exists():
        return []

    conversations = []
    for blob in container.list_blobs(name_starts_with=prefix, include=["metadata"]):
        metadata = blob.metadata or {}
        conversations.append(
            {
                "conversation_id": blob.name[len(prefix):].removesuffix(".json"),
                "title": metadata.get("title", "Untitled"),
                "updated_at": metadata.get("updated_at", ""),
                "messages": int(metadata.get("messages", 0)),
            }
        )
    return sorted(conversations, key=lambda c: c["updated_at"], reverse=True)


def list_users() -> list[str]:
    """Top-level folders in the container."""
    container = container_client(BlobLocation(account_url(), CONTAINER, ""))
    if not container.exists():
        return []
    return sorted({
        blob.name.split("/", 1)[0] for blob in container.list_blobs() if "/" in blob.name
    })


def delete(user: str, conversation_id: str) -> None:
    blob_client(_location(user, _blob_name(conversation_id))).delete_blob()


def title_from(text: str) -> str:
    """First user message, trimmed — good enough, and costs no LLM call."""
    cleaned = " ".join(text.split())
    return cleaned[:MAX_TITLE] + ("…" if len(cleaned) > MAX_TITLE else "")
