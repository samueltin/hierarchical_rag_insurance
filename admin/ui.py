"""
Document Admin Console.

    streamlit run admin/ui.py --server.port 8502

Talks to the admin API (admin/api.py), never to the pipeline or the table
directly, so the console stays a thin client.
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("ADMIN_API_URL", "http://localhost:8100")
REQUEST_TIMEOUT = 120

st.set_page_config(page_title="Doc Admin Console", page_icon="🗂️", layout="wide")

STATUS_ICON = {
    "uploaded": "⚪ uploaded",
    "running": "🔵 running",
    "succeeded": "🟢 succeeded",
    "failed": "🔴 failed",
}


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def api_error(response, path: str) -> str:
    """
    Turn a failed response into something diagnosable.

    FastAPI answers an unmatched route with a bare "Not Found", which reads as
    "your document is missing" when it actually means the API does not have
    that endpoint — usually a server running older code than the console.
    """
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    if response.status_code == 404 and detail == "Not Found":
        return (f"{API_URL} has no endpoint {path} (HTTP 404). "
                "The API is probably running older code than this console — restart it.")
    return f"HTTP {response.status_code} from {path}: {detail}"


def api_get(path: str, **params):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_stages() -> dict:
    return api_get("/stages")


def fetch_documents(doc_type: str | None = None) -> list[dict]:
    return api_get("/documents", **({"doc_type": doc_type} if doc_type else {}))


def upload_document(file, doc_type: str, version: str, recipe: dict) -> dict:
    response = requests.post(
        f"{API_URL}/documents/upload",
        files={"file": (file.name, file.getvalue(), "application/pdf")},
        data={"doc_type": doc_type, "version": version, **recipe},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(api_error(response, "/documents/upload"))
    return response.json()


def patch_document(doc_type: str, document_id: str, changes: dict) -> dict:
    response = requests.patch(
        f"{API_URL}/documents/{doc_type}/{document_id}", json=changes, timeout=REQUEST_TIMEOUT
    )
    if response.status_code >= 400:
        raise RuntimeError(api_error(response, f"/documents/{doc_type}/{document_id}"))
    return response.json()


def trigger_ingest(doc_type: str, document_id: str) -> dict:
    path = f"/documents/{doc_type}/{document_id}/ingest"
    response = requests.post(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
    if response.status_code >= 400:
        raise RuntimeError(api_error(response, path))
    return response.json()


def delete_document(doc_type: str, document_id: str, delete_blob: bool) -> None:
    response = requests.delete(
        f"{API_URL}/documents/{doc_type}/{document_id}",
        params={"delete_blob": delete_blob},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def recipe_inputs(stages: dict, current: dict, key_prefix: str) -> dict:
    """Four dropdowns, one per pipeline stage, driven by what the registry offers."""
    recipe = {}
    columns = st.columns(4)
    for column, (stage, choices) in zip(columns, stages["stages"].items()):
        with column:
            default = current.get(stage, stages["defaults"][stage])
            index = choices.index(default) if default in choices else 0
            recipe[stage] = st.selectbox(
                stage.replace("_", " ").title(),
                choices,
                index=index,
                key=f"{key_prefix}_{stage}",
            )
    return recipe


def documents_panel(documents: list[dict], stages: dict) -> None:
    if not documents:
        st.info("No documents registered yet. Use the Upload tab to add one.")
        return

    st.dataframe(
        [
            {
                "filename": d["filename"],
                "type": d["doc_type"],
                "version": d["version"] or "—",
                "status": STATUS_ICON.get(d["status"], d["status"]),
                "parents": d["parent_count"],
                "chunks": d["chunk_count"],
                "preprocessor": d["preprocessor"],
                "chunker": d["doc_chunker"],
                "last ingested": d["last_ingested_at"] or "never",
            }
            for d in documents
        ],
        width="stretch",
        hide_index=True,
    )

    failed = [d for d in documents if d["status"] == "failed"]
    for document in failed:
        st.error(f"**{document['filename']}** failed at "
                 f"{document['stage_reached'] or 'unknown stage'}: {document['error']}")


def edit_panel(documents: list[dict], stages: dict) -> None:
    """Edit a document's metadata and its pipeline recipe."""
    if not documents:
        return

    labels = [f"{d['filename']}  ({d['doc_type']})" for d in documents]
    selected = st.selectbox("Document", labels, key="edit_select")
    document = documents[labels.index(selected)]

    st.caption(f"id `{document['id']}` · {document['size_bytes']:,} bytes · "
               f"sha256 `{document['content_hash'][:16]}…`")
    st.caption(f"[source blob]({document['source_url']})")

    with st.form("edit_form"):
        left, right = st.columns(2)
        with left:
            doc_type = st.text_input("Document type", value=document["doc_type"])
        with right:
            version = st.text_input("Version", value=document["version"])

        st.markdown("**Pipeline recipe**")
        recipe = recipe_inputs(stages, document, key_prefix="edit")

        if st.form_submit_button("Save changes", type="primary"):
            changes = {"doc_type": doc_type, "version": version, **recipe}
            changed = {k: v for k, v in changes.items() if v != document.get(k)}
            if not changed:
                st.info("Nothing changed.")
            else:
                try:
                    patch_document(document["doc_type"], document["id"], changed)
                    st.success(f"Updated: {', '.join(sorted(changed))}")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))

    with st.expander("Danger zone"):
        also_blob = st.checkbox("Also delete the source file from blob storage")
        if st.button("Deregister document", type="secondary"):
            delete_document(document["doc_type"], document["id"], also_blob)
            st.success(f"Deregistered {document['filename']}")
            st.rerun()


def ingest_panel(documents: list[dict]) -> None:
    """Start a pipeline run. Runs take minutes, so this only kicks it off."""
    st.subheader("Run ingestion")
    if not documents:
        return

    runnable = [d for d in documents if d["status"] != "running"]
    running = [d for d in documents if d["status"] == "running"]

    for document in running:
        st.info(f"**{document['filename']}** is running — "
                f"stage: {document['stage_reached'] or 'starting'}. Hit Refresh for progress.")

    if not runnable:
        return

    labels = [
        f"{d['filename']}  ·  {STATUS_ICON.get(d['status'], d['status'])}" for d in runnable
    ]
    choice = st.selectbox("Document", labels, key="ingest_select")
    document = runnable[labels.index(choice)]
    st.caption("Recipe: " + ", ".join(
        f"{stage.replace('_', ' ')} = **{document[stage]}**"
        for stage in ("doc_cracker", "preprocessor", "doc_chunker", "embedding_encoder")
    ))

    if st.button("▶ Run ingestion pipeline", type="primary"):
        try:
            trigger_ingest(document["doc_type"], document["id"])
            st.success(f"Started ingesting **{document['filename']}**. "
                       "It runs in the background — hit Refresh to follow progress.")
        except RuntimeError as exc:
            st.error(str(exc))


def upload_panel(stages: dict) -> None:
    file = st.file_uploader("PDF to ingest", type=["pdf"])

    left, right = st.columns(2)
    with left:
        doc_type = st.text_input("Document type", value="policy", key="upload_doc_type")
    with right:
        version = st.text_input("Version", value="", key="upload_version")

    st.markdown("**Pipeline recipe**")
    st.caption("Which implementation runs at each stage. Defaults come from the registry.")
    recipe = recipe_inputs(stages, stages["defaults"], key_prefix="upload")

    if st.button("Upload and register", type="primary", disabled=file is None):
        try:
            result = upload_document(file, doc_type, version, recipe)
            verb = "Replaced" if result["replaced"] else "Registered"
            st.success(f"{verb} **{result['document']['filename']}** "
                       f"({result['document']['size_bytes']:,} bytes)")
            st.caption("Status is `uploaded` — run the ingestion pipeline to index it.")
        except RuntimeError as exc:
            st.error(str(exc))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("🗂️ Document Admin Console")

try:
    stages = fetch_stages()
except requests.RequestException:
    st.error(f"Cannot reach the admin API at {API_URL}. Start it with:\n\n"
             "`uvicorn admin.api:app --port 8100`")
    st.stop()

with st.sidebar:
    st.subheader("Filter")
    documents_all = fetch_documents()
    types = sorted({d["doc_type"] for d in documents_all})
    chosen = st.selectbox("Document type", ["All"] + types)
    st.caption(f"API: {API_URL}")
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

documents = documents_all if chosen == "All" else [d for d in documents_all if d["doc_type"] == chosen]

listing, editing, uploading = st.tabs(
    [f"Documents ({len(documents)})", "Edit", "Upload"]
)
with listing:
    documents_panel(documents, stages)
    st.divider()
    ingest_panel(documents)
with editing:
    edit_panel(documents, stages)
with uploading:
    upload_panel(stages)
