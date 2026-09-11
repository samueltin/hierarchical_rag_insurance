"""
Document admin API.

    uvicorn admin.api:app --reload --port 8100

Endpoints:
    GET    /health                      liveness
    GET    /stages                      stage implementations available, for the console's dropdowns
    GET    /documents                   list registered documents (optionally by doc_type)
    GET    /documents/{doc_type}/{id}   one document
    POST   /documents/upload            upload a file and register it
    PATCH  /documents/{doc_type}/{id}   edit metadata or the pipeline recipe
    DELETE /documents/{doc_type}/{id}   deregister (optionally delete the blob)
    POST   /documents/{doc_type}/{id}/ingest   start a pipeline run (202, poll for status)

A run takes minutes, so ingestion is accepted and executed in the background;
callers poll GET /documents/... for `status` and `stage_reached`.
"""

import hashlib
import os
import uuid

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from storage import BlobLocation, account_url, blob_client, upload_bytes
from pipeline import registry
from .repository import AlreadyRunning, DocumentRecord, Status, list_documents, safe_key
from . import repository, runner

load_dotenv()

INGESTION_CONTAINER = os.getenv("INGESTION_CONTAINER", "ingestion-pipeline")
SOURCE_PREFIX = os.getenv("SOURCE_DOCUMENT_PREFIX", "source-documents")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
ALLOWED_EXTENSIONS = {".pdf"}

app = FastAPI(
    title="Document Admin Console API",
    description="Register, configure and ingest policy documents",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DocumentModel(BaseModel):
    id: str
    filename: str
    doc_type: str
    version: str
    source_url: str
    content_hash: str
    size_bytes: int
    doc_cracker: str
    preprocessor: str
    doc_chunker: str
    embedding_encoder: str
    status: str
    stage_reached: str
    error: str
    chunk_count: int
    parent_count: int
    index_name: str
    embedding_model: str
    last_ingested_at: str
    created_at: str
    updated_at: str

    @classmethod
    def of(cls, record: DocumentRecord) -> "DocumentModel":
        return cls(**{f: getattr(record, f) for f in cls.model_fields})


class DocumentPatch(BaseModel):
    """Editable fields. Everything else is set by upload or by an ingestion run."""

    doc_type: str | None = None
    version: str | None = None
    doc_cracker: str | None = None
    preprocessor: str | None = None
    doc_chunker: str | None = None
    embedding_encoder: str | None = None


class UploadResponse(BaseModel):
    document: DocumentModel
    replaced: bool = Field(
        description="True when this upload overwrote an existing document's file"
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "table": repository.TABLE_NAME, "container": INGESTION_CONTAINER}


@app.get("/stages")
def stages() -> dict:
    """Available implementation per stage, plus the defaults a new document gets."""
    return {"stages": registry.options(), "defaults": registry.DEFAULT_RECIPE}


@app.get("/documents", response_model=list[DocumentModel])
def documents(doc_type: str | None = Query(None)) -> list[DocumentModel]:
    return [DocumentModel.of(r) for r in list_documents(doc_type)]


@app.get("/documents/{doc_type}/{document_id}", response_model=DocumentModel)
def document(doc_type: str, document_id: str) -> DocumentModel:
    record = repository.get(doc_type, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No document {document_id!r}")
    return DocumentModel.of(record)


@app.post("/documents/upload", response_model=UploadResponse, status_code=201)
async def upload(
    file: UploadFile = File(...),
    doc_type: str = Form("policy"),
    version: str | None = Form(None),
    doc_cracker: str | None = Form(None),
    preprocessor: str | None = Form(None),
    doc_chunker: str | None = Form(None),
    embedding_encoder: str | None = Form(None),
) -> UploadResponse:
    try:
        # Use only the basename: a browser may submit a full path, and a
        # filename is about to become a blob path segment.
        filename = safe_key(os.path.basename(file.filename or ""))
        doc_type = safe_key(doc_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Re-uploading a revised file must not silently reset the recipe an admin
    # chose earlier, so an omitted stage keeps the registered value.
    existing = next(
        (r for r in list_documents(doc_type) if r.filename == filename), None
    )
    submitted = {
        "doc_cracker": doc_cracker,
        "preprocessor": preprocessor,
        "doc_chunker": doc_chunker,
        "embedding_encoder": embedding_encoder,
    }
    recipe = {
        stage: name
        or (getattr(existing, stage) if existing else registry.DEFAULT_RECIPE[stage])
        for stage, name in submitted.items()
    }
    if version is None:
        version = existing.version if existing else ""

    try:
        registry.validate_recipe(recipe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"{extension or 'file'} not supported. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File is empty")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File exceeds the {MAX_UPLOAD_MB}MB limit"
        )

    location = BlobLocation(
        account_url(), INGESTION_CONTAINER, f"{SOURCE_PREFIX.strip('/')}/{filename}"
    )
    upload_bytes(location, data, content_type="application/pdf")

    # One row per filename within a doc_type: re-uploading updates in place
    # rather than leaving an orphan row pointing at an overwritten blob.
    record = existing or DocumentRecord(id=str(uuid.uuid4()), filename=filename)
    record.doc_type = doc_type
    record.version = version
    record.source_url = location.url
    record.content_hash = hashlib.sha256(data).hexdigest()
    record.size_bytes = len(data)
    for stage, name in recipe.items():
        setattr(record, stage, name)
    record.status = Status.UPLOADED
    record.error = ""
    record.stage_reached = ""
    repository.upsert(record)

    return UploadResponse(document=DocumentModel.of(record), replaced=existing is not None)


@app.patch("/documents/{doc_type}/{document_id}", response_model=DocumentModel)
def patch(doc_type: str, document_id: str, changes: DocumentPatch) -> DocumentModel:
    record = repository.get(doc_type, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No document {document_id!r}")

    fields = changes.model_dump(exclude_none=True)
    if not fields:
        return DocumentModel.of(record)

    recipe = {**record.recipe, **{k: v for k, v in fields.items() if k in registry.STAGE_ORDER}}
    try:
        registry.validate_recipe(recipe)
        if "doc_type" in fields:
            fields["doc_type"] = safe_key(fields["doc_type"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # doc_type is the PartitionKey, so changing it means a new row, not an update.
    new_doc_type = fields.pop("doc_type", None)
    if new_doc_type and new_doc_type != doc_type:
        for key, value in fields.items():
            setattr(record, key, value)
        record.doc_type = new_doc_type
        repository.upsert(record)
        repository.delete(doc_type, document_id)
        return DocumentModel.of(record)

    return DocumentModel.of(repository.update(doc_type, document_id, **fields))


@app.delete("/documents/{doc_type}/{document_id}", status_code=204)
def deregister(doc_type: str, document_id: str, delete_blob: bool = Query(False)) -> None:
    record = repository.get(doc_type, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No document {document_id!r}")
    if delete_blob and record.source_url:
        from storage import parse_blob_url

        blob_client(parse_blob_url(record.source_url)).delete_blob()
    repository.delete(doc_type, document_id)


class IngestResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    poll: str = Field(description="Poll this URL for status and stage_reached")


@app.post(
    "/documents/{doc_type}/{document_id}/ingest",
    response_model=IngestResponse,
    status_code=202,
)
def ingest(doc_type: str, document_id: str, background: BackgroundTasks) -> IngestResponse:
    """
    Start a pipeline run and return immediately.

    The row is claimed synchronously, before the work is scheduled: doing it
    inside the background task would leave a window where two rapid triggers
    both pass the "already running" check.
    """
    record = repository.get(doc_type, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No document {document_id!r}")
    if not record.source_url:
        raise HTTPException(status_code=409, detail="Document has no file; upload one first")

    try:
        repository.claim(doc_type, document_id)
    except AlreadyRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    background.add_task(runner.run_claimed, doc_type, document_id)
    return IngestResponse(
        status="accepted",
        document_id=document_id,
        filename=record.filename,
        poll=f"/documents/{doc_type}/{document_id}",
    )
