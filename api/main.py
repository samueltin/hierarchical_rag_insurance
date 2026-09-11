"""
FastAPI wrapper around rag_retriever.answer_question.

    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET    /health                                liveness plus which index is being queried
    GET    /documents                             document names available in the index
    GET    /chunk?parent_path=...                  one parent chunk's text, for citations
    GET    /users                                 users with stored chat history
    GET    /users/{user}/conversations            conversation list (titles only)
    POST   /users/{user}/conversations            start a new conversation
    GET    /users/{user}/conversations/{id}       full transcript
    DELETE /users/{user}/conversations/{id}       delete a conversation
    POST   /ask                                   answer a question, with its sources

When /ask carries a conversation_id, the server appends both the question and
the answer to the transcript — the client never writes history, so two open
tabs cannot diverge.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from chat import groundedness, history
from rag_retriever import SEARCH_INDEX_NAME, answer_question, breadcrumb_of
from storage import account_url, download_json, parse_blob_url

load_dotenv()

app = FastAPI(
    title="Hierarchical RAG — Insurance Policy QA",
    description="Question answering over policy documents indexed by the ingestion pipeline",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    document_name: str | None = Field(
        None, description="Restrict the search to a single ingested document"
    )
    user: str | None = Field(None, description="Whose history to append to")
    conversation_id: str | None = Field(
        None, description="Conversation to append this exchange to"
    )


class SourceModel(BaseModel):
    """A policy section that fed the answer. Ordered best match first."""

    section: str
    document_name: str
    parent_id: str
    parent_path: str
    content: str
    chars: int
    score: float = 0.0
    hits: int = 0


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceModel]
    conversation_id: str | None = None
    search_query: str = Field(
        "", description="What retrieval searched on; differs when a follow-up was condensed"
    )
    groundedness: dict = Field(
        default_factory=dict,
        description="Content Safety groundedness verdict; checked=False when not run",
    )


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    updated_at: str
    messages: int


class MessageModel(BaseModel):
    role: str
    content: str
    ts: str
    document_name: str = ""
    search_query: str = ""
    sources: list[dict] = []
    groundedness: dict = {}


class ConversationModel(BaseModel):
    conversation_id: str
    user: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageModel]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index": SEARCH_INDEX_NAME,
        "chat_deployment": os.getenv("CHAT_DEPLOYMENT"),
        "embedding_deployment": os.getenv("EMBEDDING_DEPLOYMENT"),
        "groundedness": {
            "configured": groundedness.is_configured(),
            "enabled": groundedness.is_enabled(),
            "sources": os.getenv("GROUNDEDNESS_SOURCES", "all"),
        },
    }


@app.get("/documents")
def documents() -> dict:
    """Document names present in the index — `document_name` is facetable."""
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient

    client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=SEARCH_INDEX_NAME,
        credential=DefaultAzureCredential(),
    )
    results = client.search(search_text="*", facets=["document_name"], top=0)
    facets = results.get_facets() or {}
    return {
        "documents": [
            {"document_name": f["value"], "chunks": f["count"]}
            for f in facets.get("document_name", [])
        ]
    }


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

DEFAULT_USERS = [u.strip() for u in os.getenv("CHAT_USERS", "user1,user2").split(",") if u.strip()]
# Turns of context sent to the condenser and the answer prompt. Kept small:
# retrieved parents already cost up to ~20K characters per turn.
HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "6"))


@app.get("/users")
def users() -> dict:
    """Configured users, plus anyone who already has stored history."""
    stored = history.list_users()
    return {"users": sorted(set(DEFAULT_USERS) | set(stored))}


@app.get("/users/{user}/conversations", response_model=list[ConversationSummary])
def conversations(user: str) -> list[ConversationSummary]:
    try:
        return [ConversationSummary(**row) for row in history.list_conversations(user)]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/users/{user}/conversations", response_model=ConversationModel, status_code=201)
def new_conversation(user: str) -> ConversationModel:
    try:
        conversation = history.create(user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _conversation_model(conversation)


@app.get("/users/{user}/conversations/{conversation_id}", response_model=ConversationModel)
def conversation(user: str, conversation_id: str) -> ConversationModel:
    found = history.load(user, conversation_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"No conversation {conversation_id!r}")
    return _conversation_model(found)


@app.delete("/users/{user}/conversations/{conversation_id}", status_code=204)
def delete_conversation(user: str, conversation_id: str) -> None:
    if history.load(user, conversation_id) is None:
        raise HTTPException(status_code=404, detail=f"No conversation {conversation_id!r}")
    history.delete(user, conversation_id)


def _conversation_model(conversation) -> ConversationModel:
    return ConversationModel(
        conversation_id=conversation.conversation_id,
        user=conversation.user,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageModel(
                role=m.role,
                content=m.content,
                ts=m.ts,
                document_name=m.document_name,
                search_query=m.search_query,
                groundedness=m.groundedness,
                sources=[vars(s) for s in m.sources],
            )
            for m in conversation.messages
        ],
    )


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    conversation = None
    if request.user and request.conversation_id:
        conversation = history.load(request.user, request.conversation_id)
        if conversation is None:
            raise HTTPException(
                status_code=404, detail=f"No conversation {request.conversation_id!r}"
            )

    # Recent turns, so a follow-up can be condensed into a standalone question
    # and the model can see what "it" refers to.
    turns = (
        [{"role": m.role, "content": m.content} for m in conversation.turns(HISTORY_TURNS)]
        if conversation is not None else []
    )

    try:
        result = answer_question(
            request.query,
            document_name=request.document_name,
            history=turns,
        )
    except Exception as exc:  # surface Azure/config failures as a clean 502
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    # Advisory check: is the answer actually supported by the sections retrieved?
    verdict = groundedness.check(
        query=request.query,
        answer=result.answer,
        source_texts=[s.content for s in result.sources],
    ).to_dict()

    if conversation is not None:
        conversation.messages.append(
            history.Message(
                role="user", content=request.query,
                document_name=request.document_name or "",
                # Kept so a bad answer can be traced to the condenser or the search.
                search_query=result.search_query if result.search_query != request.query else "",
            )
        )
        conversation.messages.append(
            history.Message(
                role="assistant", content=result.answer,
                document_name=request.document_name or "",
                sources=[
                    history.SourceRef(
                        section=s.section,
                        document_name=s.document_name,
                        parent_path=s.parent_path,
                        score=s.score,
                        hits=s.hits,
                    )
                    for s in result.sources
                ],
                groundedness=verdict,
            )
        )
        if len(conversation.messages) <= 2:
            conversation.title = history.title_from(request.query)
        try:
            history.save(conversation)
        except history.ConversationConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return AskResponse(
        query=request.query,
        answer=result.answer,
        sources=[SourceModel(**vars(source)) for source in result.sources],
        conversation_id=request.conversation_id,
        search_query=result.search_query,
        groundedness=verdict,
    )


# ---------------------------------------------------------------------------
# Chunk lookup
# ---------------------------------------------------------------------------

INGESTION_CONTAINER = os.getenv("INGESTION_CONTAINER", "ingestion-pipeline")
PARENT_CHUNK_PREFIX = os.getenv("PARENT_CHUNK_PREFIX", "parent-chunks")


class ChunkModel(BaseModel):
    parent_path: str
    document_name: str
    section: str
    content: str
    chars: int


@app.get("/chunk", response_model=ChunkModel)
def chunk(parent_path: str = Query(..., description="Blob URL of a parent chunk")) -> ChunkModel:
    """
    Fetch one parent chunk's text.

    Transcripts store citations as references rather than text, so the UI reads
    the wording from here when it needs to show it. The path comes from a
    client, so it is checked against this account's parent-chunk prefix — an
    unchecked blob URL would let a caller read any blob the service can reach.
    """
    try:
        location = parse_blob_url(parent_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    expected_prefix = f"{PARENT_CHUNK_PREFIX.strip('/')}/"
    if (
        location.account_url != account_url()
        or location.container != INGESTION_CONTAINER
        or not location.name.startswith(expected_prefix)
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Not a parent chunk in {INGESTION_CONTAINER}/{expected_prefix}",
        )

    try:
        payload = download_json(location)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Chunk not found: {type(exc).__name__}") from exc

    content = payload.get("content", "")
    return ChunkModel(
        parent_path=parent_path,
        document_name=payload.get("document_name", ""),
        section=breadcrumb_of(payload),
        content=content,
        chars=len(content),
    )
