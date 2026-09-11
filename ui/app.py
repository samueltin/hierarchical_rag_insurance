"""
Streamlit chat UI for the policy QA API.

    streamlit run ui/app.py

Talks to the FastAPI server (api/main.py) rather than importing rag_retriever
directly, so the UI stays a thin client. The server owns conversation history:
this page never writes a transcript, it just renders what the API returns.
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180

st.set_page_config(page_title="Policy Assistant", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def api_get(path: str, **params):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_documents() -> list[dict]:
    try:
        return api_get("/documents")["documents"]
    except requests.RequestException:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def fetch_users() -> list[str]:
    return api_get("/users")["users"]


@st.cache_data(ttl=600, show_spinner=False)
def fetch_chunk(parent_path: str) -> dict:
    """
    Chunk text for a citation.

    Transcripts store citations as references, not text, so the wording is
    fetched here and cached — the same section is often cited across turns.
    """
    try:
        return api_get("/chunk", parent_path=parent_path)
    except requests.RequestException:
        return {}


def fetch_conversations(user: str) -> list[dict]:
    return api_get(f"/users/{user}/conversations")


def fetch_conversation(user: str, conversation_id: str) -> dict:
    return api_get(f"/users/{user}/conversations/{conversation_id}")


def new_conversation(user: str) -> dict:
    response = requests.post(f"{API_URL}/users/{user}/conversations", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def delete_conversation(user: str, conversation_id: str) -> None:
    requests.delete(
        f"{API_URL}/users/{user}/conversations/{conversation_id}", timeout=REQUEST_TIMEOUT
    ).raise_for_status()


def ask(query: str, document_name: str | None, user: str, conversation_id: str) -> dict:
    response = requests.post(
        f"{API_URL}/ask",
        json={
            "query": query,
            "document_name": document_name,
            "user": user,
            "conversation_id": conversation_id,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def source_text(source: dict) -> str:
    """
    The chunk wording for a citation.

    A fresh answer carries it inline; a citation replayed from a stored
    transcript carries only a parent_path, so it is fetched on demand.
    """
    if source.get("content"):
        return source["content"]
    if source.get("parent_path"):
        return fetch_chunk(source["parent_path"]).get("content", "")
    return ""


def render_sources(sources: list[dict], key: str) -> None:
    """The best-matching section beside the answer, the rest behind an expander."""
    if not sources:
        st.info("No sources returned.")
        return

    best = sources[0]           # sources come back best match first
    content = source_text(best)
    st.caption("Top matching section")
    st.markdown(f"**{best.get('section') or '(no heading)'}**")
    detail = [best.get("document_name", "")]
    if content:
        detail.append(f"{len(content):,} chars")
    if best.get("hits"):
        detail.append(f"{best['hits']} matching chunk{'s' if best['hits'] > 1 else ''}")
    if best.get("score"):
        detail.append(f"score {best['score']:.4f}")
    st.caption(" · ".join(d for d in detail if d))
    if content:
        st.text_area("Chunk text", value=content, height=240,
                     label_visibility="collapsed", key=f"chunk_{key}")
    else:
        st.caption("Chunk text unavailable.")
    if best.get("parent_path"):
        st.caption(f"[parent chunk blob]({best['parent_path']})")

    if len(sources) > 1:
        with st.expander(f"{len(sources) - 1} other section(s) retrieved"):
            for other in sources[1:]:
                st.markdown(f"**{other.get('section') or '(no heading)'}**")
                st.caption(" · ".join(d for d in [
                    other.get("document_name", ""),
                    f"{other['hits']} chunk(s)" if other.get("hits") else "",
                    f"score {other['score']:.4f}" if other.get("score") else "",
                ] if d))
                other_text = source_text(other)
                if other_text:
                    st.text(other_text[:500] + ("…" if len(other_text) > 500 else ""))
                st.divider()


def render_groundedness(verdict: dict) -> None:
    """
    Show whether Content Safety judged the answer supported by its sources.

    Advisory, and shown as such: an unchecked answer is not a failed one, and a
    flagged answer is a prompt to read the cited section, not proof of an error.
    """
    if not verdict:
        return

    if not verdict.get("checked"):
        st.caption(f"⚪ Groundedness not checked — {verdict.get('detail', 'unavailable')}")
        return

    grounded = verdict.get("grounded_percentage", 0.0)
    sources_used = verdict.get("sources_used", 0)
    suffix = f"against {sources_used} retrieved section{'s' if sources_used != 1 else ''}"

    if verdict.get("ungrounded"):
        st.warning(f"⚠️ {grounded:.0f}% grounded {suffix} — "
                   "some statements were not found in the policy text.")
        spans = verdict.get("ungrounded_text") or []
        if spans:
            with st.expander(f"{len(spans)} unsupported statement(s)"):
                for span in spans:
                    st.markdown(f"> {span}")
    else:
        st.success(f"✅ {grounded:.0f}% grounded {suffix}.")


def render_message(message: dict, index: int) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            # Present when a follow-up was rewritten before searching — shows why
            # a given set of sections came back.
            if message.get("search_query"):
                st.caption(f"🔍 searched as: _{message['search_query']}_")
            return
        answer_col, source_col = st.columns([3, 2], gap="large")
        with answer_col:
            st.markdown(message["content"])
            render_groundedness(message.get("groundedness", {}))
        with source_col:
            render_sources(message.get("sources", []), key=str(index))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("📄 Policy Assistant")

try:
    users = fetch_users()
except requests.RequestException:
    st.error(f"Cannot reach the API at {API_URL}. Start it with "
             "`uvicorn api.main:app --port 8000`")
    st.stop()

with st.sidebar:
    st.subheader("User")
    user = st.selectbox("Signed in as", users, key="user")
    st.caption("A demo switcher, not authentication — each user sees only their own chats.")

    if st.button("＋ New chat", width="stretch"):
        created = new_conversation(user)
        st.session_state["conversation_id"] = created["conversation_id"]
        st.rerun()

    st.subheader("Chats")
    conversations = fetch_conversations(user)
    if not conversations:
        st.caption("No chats yet — start one above.")
        selected_id = st.session_state.get("conversation_id")
    else:
        labels = [f"{c['title']}  ({c['messages'] // 2} turns)" for c in conversations]
        ids = [c["conversation_id"] for c in conversations]
        current = st.session_state.get("conversation_id")
        index = ids.index(current) if current in ids else 0
        chosen = st.radio("Conversation", labels, index=index, label_visibility="collapsed")
        selected_id = ids[labels.index(chosen)]
        st.session_state["conversation_id"] = selected_id

    st.divider()
    st.subheader("Scope")
    documents = fetch_documents()
    doc_labels = ["All documents"] + [f"{d['document_name']} ({d['chunks']})" for d in documents]
    choice = st.selectbox("Search within", doc_labels)
    selected_document = (
        None if choice == "All documents" else documents[doc_labels.index(choice) - 1]["document_name"]
    )
    st.caption("Applies to the next question only — a chat can span documents.")

    st.divider()
    if selected_id and st.button("Delete this chat"):
        delete_conversation(user, selected_id)
        st.session_state.pop("conversation_id", None)
        st.rerun()
    st.caption(f"API: {API_URL}")

# Transcript comes from the server, so it survives reloads and user switches.
if not selected_id:
    st.info("Start a new chat from the sidebar.")
    st.stop()

transcript = fetch_conversation(user, selected_id)
for index, message in enumerate(transcript["messages"]):
    render_message(message, index)

if query := st.chat_input("Ask about the policies…"):
    with st.chat_message("user"):
        st.markdown(query)
    with st.spinner("Searching the policies…"):
        try:
            ask(query, selected_document, user, selected_id)
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
            st.stop()
    st.rerun()
