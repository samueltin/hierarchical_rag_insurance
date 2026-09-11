"""
Run a registered document's pipeline and record the outcome on its row.

    python -m admin.runner <document-id>          ingest one registered document
    python -m admin.runner --all                  ingest everything not yet succeeded
    python -m admin.runner --list                 show what is registered

This is the thin layer between the document table and pipeline.runner: it turns
a row into a recipe, tracks progress on the row while the run is in flight, and
writes the result back. The pipeline itself knows nothing about any of that.
"""

import sys

from dotenv import load_dotenv

from pipeline import runner as pipeline_runner
from pipeline.registry import STAGE_ORDER

from . import repository
from .repository import AlreadyRunning, DocumentRecord, Status, utcnow

load_dotenv()


def stage_options(record: DocumentRecord) -> dict[str, dict]:
    """
    Per-stage constructor arguments for this document.

    The cracker is the expensive stage — two minutes of Document Intelligence
    for a 34-page booklet — so skip it when the file has not changed since the
    last successful run. When the content hash differs, the existing markdown
    belongs to the old file and must be re-extracted.
    """
    unchanged = bool(record.ingested_content_hash) and (
        record.ingested_content_hash == record.content_hash
    )
    return {
        "doc_cracker": {"skip_if_exists": unchanged},
        "preprocessor": {"skip_if_exists": False},
    }


def ingest(doc_type: str, document_id: str) -> DocumentRecord:
    """
    Run the pipeline for one document, start to finish.

    Raises AlreadyRunning if a run is already in flight; every other failure is
    recorded on the row rather than raised, since the caller is usually a
    background task with nowhere to report to.
    """
    repository.claim(doc_type, document_id)
    return run_claimed(doc_type, document_id)


def run_claimed(doc_type: str, document_id: str) -> DocumentRecord:
    """
    Run the pipeline for a document already marked running.

    Split out so an HTTP caller can claim the row synchronously (rejecting a
    duplicate trigger with 409) and do the work in the background.
    """
    record = repository.get(doc_type, document_id)
    if record is None:
        raise KeyError(f"No document {document_id!r} of type {doc_type!r}")
    if not record.source_url:
        return _fail(record, "Document has no source_url — upload the file first")

    def on_stage(stage: str, context: dict) -> None:
        repository.update(doc_type, document_id, stage_reached=stage)

    result = pipeline_runner.run(
        record.recipe,
        record.source_url,
        stage_options=stage_options(record),
        on_stage=on_stage,
    )

    if not result.succeeded:
        where = result.failed_stage or result.stage_reached or "start"
        return _fail(record, result.error or "Pipeline did not complete", stage=where)

    context = result.context
    return repository.update(
        doc_type,
        document_id,
        status=Status.SUCCEEDED,
        stage_reached=STAGE_ORDER[-1],
        error="",
        chunk_count=context.get("chunk_count", 0),
        parent_count=context.get("parent_count", 0),
        index_name=context.get("index_name", ""),
        embedding_model=context.get("embedding_model", ""),
        ingested_content_hash=record.content_hash,
        last_ingested_at=utcnow(),
    )


def _fail(record: DocumentRecord, message: str, stage: str = "") -> DocumentRecord:
    return repository.update(
        record.doc_type,
        record.id,
        status=Status.FAILED,
        error=message[:1024],
        stage_reached=stage,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_documents() -> None:
    rows = repository.list_documents()
    if not rows:
        print("No documents registered.")
        return
    print(f"{'id':38} {'filename':36} {'status':10} {'parents/chunks':>15}")
    for r in rows:
        print(f"{r.id:38} {r.filename:36} {r.status:10} "
              f"{str(r.parent_count) + '/' + str(r.chunk_count):>15}")


def _run_one(record: DocumentRecord) -> bool:
    print(f"\n=== {record.filename} ({record.doc_type}) ===")
    print("    recipe: " + ", ".join(f"{s}={record.recipe[s]}" for s in STAGE_ORDER))
    try:
        done = ingest(record.doc_type, record.id)
    except AlreadyRunning as exc:
        print(f"    skipped: {exc}")
        return False
    if done.status == Status.SUCCEEDED:
        print(f"    succeeded: {done.parent_count} parents / {done.chunk_count} chunks "
              f"-> {done.index_name}")
        return True
    print(f"    FAILED at {done.stage_reached}: {done.error}")
    return False


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--list":
        _print_documents()
        return 0

    if argv[0] == "--all":
        pending = [r for r in repository.list_documents() if r.status != Status.SUCCEEDED]
        if not pending:
            print("Nothing to ingest — every document has succeeded.")
            return 0
        return 0 if all([_run_one(r) for r in pending]) else 1

    record = repository.find(argv[0])
    if record is None:
        print(f"No document with id {argv[0]!r}. Use --list to see what is registered.")
        return 2
    return 0 if _run_one(record) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
