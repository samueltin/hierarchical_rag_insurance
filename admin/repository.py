"""
Document registry backed by Azure Table Storage.

One row per document: what it is, which pipeline recipe ingests it, and how the
last ingestion went. This is the only module that knows the store is Tables —
swapping to Cosmos or Postgres means rewriting this file and nothing else.

Keys: PartitionKey = doc_type, RowKey = id.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import StrEnum
import os
import re

from azure.core import MatchConditions
from azure.core.exceptions import ResourceModifiedError, ResourceNotFoundError
from azure.data.tables import TableServiceClient, UpdateMode

from pipeline.registry import DEFAULT_RECIPE, STAGE_ORDER

TABLE_NAME = os.getenv("DOCUMENT_TABLE_NAME", "documents")

# Azure Table keys cannot contain these, and a filename must not escape its prefix.
_UNSAFE_KEY_CHARS = re.compile(r"[/\\#?\x00-\x1f]")


class AlreadyRunning(RuntimeError):
    """Raised when a document already has a pipeline run in flight."""


class Status(StrEnum):
    """Where a document is in its ingestion lifecycle."""

    UPLOADED = "uploaded"      # file is in blob storage, never ingested
    RUNNING = "running"        # a pipeline run is in flight
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DocumentRecord:
    """One registered document."""

    id: str
    filename: str
    doc_type: str = "policy"
    version: str = ""
    source_url: str = ""
    content_hash: str = ""
    size_bytes: int = 0

    # Pipeline recipe — class names resolved through pipeline.registry
    doc_cracker: str = DEFAULT_RECIPE["doc_cracker"]
    preprocessor: str = DEFAULT_RECIPE["preprocessor"]
    doc_chunker: str = DEFAULT_RECIPE["doc_chunker"]
    embedding_encoder: str = DEFAULT_RECIPE["embedding_encoder"]

    # Last ingestion outcome
    ingested_content_hash: str = ""   # content_hash as of the last successful run
    status: str = Status.UPLOADED
    stage_reached: str = ""
    error: str = ""
    chunk_count: int = 0
    parent_count: int = 0
    index_name: str = ""
    embedding_model: str = ""
    last_ingested_at: str = ""

    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    @property
    def recipe(self) -> dict[str, str]:
        return {stage: getattr(self, stage) for stage in STAGE_ORDER}

    def to_entity(self) -> dict:
        entity = asdict(self)
        entity["PartitionKey"] = self.doc_type
        entity["RowKey"] = self.id
        return entity

    @classmethod
    def from_entity(cls, entity: dict) -> "DocumentRecord":
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in entity.items() if k in known}
        values.setdefault("doc_type", entity.get("PartitionKey", ""))
        values.setdefault("id", entity.get("RowKey", ""))
        return cls(**values)


def safe_key(value: str) -> str:
    """
    Validate a value used as a table key or blob path segment.

    Rejects rather than sanitises: silently rewriting "../evil.pdf" to
    "..evil.pdf" would store the file under a name the admin never chose.
    """
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError("Value must not be empty")
    if _UNSAFE_KEY_CHARS.search(candidate):
        raise ValueError(
            f"{value!r} contains characters not allowed in a name: / \\ # ? or control characters"
        )
    if candidate in (".", "..") or candidate.startswith(".."):
        raise ValueError(f"{value!r} is not a valid name")
    return candidate


# ---------------------------------------------------------------------------
# Table access
# ---------------------------------------------------------------------------

_table_client = None


def table():
    """Table client, created (and the table created) on first use."""
    global _table_client
    if _table_client is None:
        connection_string = os.getenv("BLOB_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("BLOB_CONNECTION_STRING is required for the document table")
        service = TableServiceClient.from_connection_string(connection_string)
        service.create_table_if_not_exists(TABLE_NAME)
        _table_client = service.get_table_client(TABLE_NAME)
    return _table_client


def upsert(record: DocumentRecord) -> DocumentRecord:
    record.updated_at = utcnow()
    table().upsert_entity(record.to_entity())
    return record


def get(doc_type: str, document_id: str) -> DocumentRecord | None:
    try:
        entity = table().get_entity(partition_key=doc_type, row_key=document_id)
    except ResourceNotFoundError:
        return None
    return DocumentRecord.from_entity(entity)


def find(document_id: str) -> DocumentRecord | None:
    """Look a document up by id alone, when the caller has no doc_type."""
    rows = table().query_entities(
        "RowKey eq @id", parameters={"id": document_id}
    )
    for entity in rows:
        return DocumentRecord.from_entity(entity)
    return None


def list_documents(doc_type: str | None = None) -> list[DocumentRecord]:
    if doc_type:
        entities = table().query_entities(
            "PartitionKey eq @doc_type", parameters={"doc_type": doc_type}
        )
    else:
        entities = table().list_entities()
    records = [DocumentRecord.from_entity(e) for e in entities]
    return sorted(records, key=lambda r: (r.doc_type, r.filename, r.version))


def update(doc_type: str, document_id: str, **fields) -> DocumentRecord:
    """Patch named fields on an existing row."""
    record = get(doc_type, document_id)
    if record is None:
        raise KeyError(f"No document {document_id!r} of type {doc_type!r}")
    unknown = set(fields) - set(record.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown field(s): {', '.join(sorted(unknown))}")
    for key, value in fields.items():
        setattr(record, key, value)
    return upsert(record)


def claim(doc_type: str, document_id: str) -> DocumentRecord:
    """
    Mark a document as running, failing if another run got there first.

    Uses the entity's ETag, so the check-and-set is atomic: two triggers
    arriving together cannot both start a pipeline for the same document.
    """
    try:
        entity = table().get_entity(partition_key=doc_type, row_key=document_id)
    except ResourceNotFoundError as exc:
        raise KeyError(f"No document {document_id!r} of type {doc_type!r}") from exc

    if entity.get("status") == Status.RUNNING:
        raise AlreadyRunning(f"{entity.get('filename', document_id)} is already being ingested")

    record = DocumentRecord.from_entity(entity)
    record.status = Status.RUNNING
    record.stage_reached = ""
    record.error = ""
    record.updated_at = utcnow()

    try:
        table().update_entity(
            record.to_entity(),
            mode=UpdateMode.MERGE,
            etag=entity.metadata["etag"],
            match_condition=MatchConditions.IfNotModified,
        )
    except ResourceModifiedError as exc:
        raise AlreadyRunning(
            f"{record.filename} was modified by another request; try again"
        ) from exc
    return record


def delete(doc_type: str, document_id: str) -> None:
    table().delete_entity(partition_key=doc_type, row_key=document_id)
