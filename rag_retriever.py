"""
rag_retriever.py

RAG query engine using the parent-child index built by the ingestion pipeline.

Query time flow:
  0. Condense a follow-up into a standalone question (multi-turn only)
  1. Embed the query via Azure OpenAI
  2. Vector search Azure AI Search  →  top matching child chunks
  3. Extract parent blob URLs from results (deduplicated)
  4. Fetch those parent JSON blobs from Azure Blob Storage
  5. Build context from parent content + section metadata
  6. Pass context + question to LLM and return the answer with its sources
"""

import os
import re
from dataclasses import dataclass, field
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.identity import DefaultAzureCredential

from storage import download_json, parse_blob_url

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
EMBEDDING_DEPLOYMENT     = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
CHAT_DEPLOYMENT          = os.getenv("CHAT_DEPLOYMENT", "gpt-4o")

SEARCH_ENDPOINT          = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX_NAME        = os.getenv("AZURE_SEARCH_INDEX_NAME", "breakdown-child-chunks")

TOP_K = 5  # number of child chunks to retrieve per query

# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve_parent_chunks(query: str, document_name: str | None = None) -> list[dict]:
    """
    Embed query -> vector search child chunks -> fetch parent blobs.
    Returns a list of parent dicts: {parent_id, content, metadata}.

    Pass `document_name` to scope the search to a single ingested document.
    """

    # 1. Embed the query
    embedder = AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=lambda: DefaultAzureCredential()
            .get_token("https://cognitiveservices.azure.com/.default").token,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    query_vector = embedder.embed_query(query)

    # 2. Vector search over child chunks in Azure AI Search
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=DefaultAzureCredential(),
    )
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=TOP_K,
        fields="content_vector",
    )
    results = list(search_client.search(
        search_text=query,           # hybrid: keyword + vector search combined
        vector_queries=[vector_query],
        filter=f"document_name eq '{document_name}'" if document_name else None,
        select=["child_id", "parent_id", "parent_path", "document_name", "content",
                "meta_h1", "meta_h2", "meta_h3", "meta_h4"],
        top=TOP_K,
    ))

    # 3. Group children by parent, adding up their scores.
    #
    #    Ranking parents by their best single child favours short chunks: a
    #    108-character cross-reference that is almost entirely query terms
    #    out-scores the 420-character section that answers the question, on both
    #    keyword and vector search. A parent whose children match repeatedly is
    #    the better bet, so the scores are summed.
    matches: dict[str, dict] = {}
    for r in results:
        # The ingestion pipeline stamps each child with its parent's blob URL, so
        # query time does not need to know how the pipeline lays out storage.
        path = r.get("parent_path")
        if not path:
            print(f"      Skipping chunk {r['child_id']}: no parent_path "
                  f"(indexed before the ingestion pipeline refactor)")
            continue
        score = r.get("@search.score", 0.0)
        match = matches.setdefault(path, {"score": 0.0, "hits": 0, "best": score})
        match["score"] += score
        match["hits"] += 1
        match["best"] = max(match["best"], score)

    ranked = sorted(matches.items(), key=lambda kv: kv[1]["score"], reverse=True)

    # 4. Fetch parent blobs from Azure Blob Storage, keeping the URL each came
    #    from so the caller can cite it.
    parents = []
    for path, match in ranked:
        parent = download_json(parse_blob_url(path))
        parent["parent_path"] = path
        parent["score"] = round(match["score"], 5)
        parent["hits"] = match["hits"]
        parents.append(parent)
    return parents


def breadcrumb_of(parent: dict) -> str:
    """"Your Cover > Section A. Roadside" from a parent chunk's heading metadata."""
    meta = parent.get("metadata", {})
    return " > ".join(
        v for v in [
            meta.get("H1", ""),
            meta.get("H2", ""),
            meta.get("H3", ""),
            meta.get("H4", ""),
        ] if v
    )


def source_ids(parents: list[dict]) -> dict[str, dict]:
    """Stable id per retrieved section — "S1", "S2" — in rank order."""
    return {f"S{i}": parent for i, parent in enumerate(parents, 1)}


def build_context(parents: list[dict], ids: dict[str, dict] | None = None) -> str:
    """
    Format parent chunks into a context block, grouped by policy document.

    The corpus holds several policies that share section names — each has its
    own "Cancellation rights", "Complaints procedure" and "General exclusions".
    Presenting the sections as one flat list invites the model to merge rules
    from different policies into a single answer with the wrong numbers, so
    every section is labelled with the document it came from.

    Each section also carries a short id, which the model returns to say which
    ones it used. Reading the ids back is exact; inferring provenance from the
    prose was not.
    """
    ids = ids or source_ids(parents)
    id_of = {id(parent): key for key, parent in ids.items()}

    by_document: dict[str, list[dict]] = {}
    for parent in parents:
        by_document.setdefault(parent.get("document_name") or "unknown", []).append(parent)

    blocks = []
    for document_name, items in by_document.items():
        lines = [f"## Policy document: {document_name}"]
        for parent in items:
            breadcrumb = breadcrumb_of(parent)
            key = id_of.get(id(parent), "?")
            lines.append(f"[{key}] {document_name} › {breadcrumb or 'General'}\n{parent['content']}")
        blocks.append("\n\n".join(lines))
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class GroundedAnswer(BaseModel):
    """What the model returns: the answer, and which sections it used."""

    answer: str = Field(
        description="The answer in markdown, drawn only from the supplied sections."
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description=(
            "Ids of the sections the answer is based on, e.g. ['S3', 'S1'], "
            "MOST IMPORTANT FIRST. The first id must be the section the answer "
            "is chiefly drawn from. Omit sections only mentioned in passing. "
            "Empty if the sections do not answer the question."
        ),
    )
    sufficient: bool = Field(
        default=True,
        description="False when the supplied sections do not answer the question.",
    )

@dataclass
class Source:
    """One parent chunk that fed the answer."""

    section: str          # heading breadcrumb, e.g. "Your Cover > Section A. Roadside"
    document_name: str
    parent_id: str
    parent_path: str      # blob URL of the parent chunk
    content: str          # the parent chunk text the answer was drawn from
    chars: int
    score: float = 0.0    # combined search score of this parent's matching chunks
    hits: int = 0         # how many of its child chunks matched
    cited: bool = False   # the answer names this section as its source
    cite_order: int = -1  # 0 = cited first in the answer, -1 = not cited

    def __str__(self) -> str:
        return (f"{self.section or '(no heading)'} "
                f"[{self.document_name}, {self.chars} chars, score {self.score:.4f}]")

    @classmethod
    def from_parent(cls, parent: dict) -> "Source":
        content = parent.get("content", "")
        return cls(
            section=breadcrumb_of(parent),
            document_name=parent.get("document_name", ""),
            parent_id=parent.get("parent_id", ""),
            parent_path=parent.get("parent_path", ""),
            content=content,
            chars=len(content),
            score=parent.get("score", 0.0),
            hits=parent.get("hits", 0),
        )


@dataclass
class RagAnswer:
    """
    An answer together with the policy sections it was drawn from.

    Stringifies to the answer text, so `print(answer)` still shows just the answer.
    """

    answer: str
    query: str = ""
    # False when the retrieved sections did not answer the question.
    sufficient: bool = True
    # What retrieval actually searched on — differs from `query` when a
    # follow-up was condensed. Recorded so a bad answer can be traced to the
    # condenser or to the search, rather than guessed at.
    search_query: str = ""
    sources: list[Source] = field(default_factory=list)

    def __str__(self) -> str:
        return self.answer

    def cited(self) -> str:
        """Answer followed by a numbered source list."""
        if not self.sources:
            return self.answer
        lines = [self.answer, "", "Sources:"]
        lines += [f"  [{i}] {s}" for i, s in enumerate(self.sources, 1)]
        return "\n".join(lines)


def generate(prompt: str) -> GroundedAnswer:
    """
    Ask for an answer plus the ids of the sections it used.

    Structured output is tried strictly first, then via tool calling, and if
    both fail the model is asked for plain prose. A schema problem should cost
    the citations, never the answer.
    """
    for method in ("json_schema", "function_calling"):
        try:
            return chat_model().with_structured_output(
                GroundedAnswer, method=method
            ).invoke(prompt)
        except Exception as exc:
            print(f"      Structured output via {method} failed: "
                  f"{type(exc).__name__}: {str(exc)[:120]}")

    print("      Falling back to an unstructured answer — citations unavailable")
    return GroundedAnswer(answer=chat_model().invoke(prompt).content, sources_used=[])


def apply_citations(result: GroundedAnswer, ids: dict[str, dict],
                    sources: list[Source]) -> list[Source]:
    """
    Mark the sections the model says it used, in the order it ranked them.

    The model is given each section with a short id and returns the ids it used,
    most important first, so provenance is read back exactly. Unknown ids are
    ignored — a model can invent "S9" — and the order it gives is kept, because
    an answer's opening source is what it is built on while later ones are
    qualifications.
    """
    by_parent_path = {
        parent.get("parent_path"): key for key, parent in ids.items()
    }
    key_of_source = {
        source.parent_path: by_parent_path.get(source.parent_path) for source in sources
    }

    position = 0
    for key in result.sources_used:
        if key not in ids:
            continue
        for source in sources:
            if key_of_source.get(source.parent_path) == key and not source.cited:
                source.cited = True
                source.cite_order = position
                position += 1
                break
    return sources


# ---------------------------------------------------------------------------
# Multi-turn support
# ---------------------------------------------------------------------------

def chat_model(temperature: float = 0) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=CHAT_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=lambda: DefaultAzureCredential()
            .get_token("https://cognitiveservices.azure.com/.default").token,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=temperature,
    )


def format_history(history: list[dict], limit: int = 6) -> str:
    """Recent turns as plain text. Only the words — never the retrieved chunks."""
    recent = history[-limit * 2:] if limit else history
    return "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '').strip()}"
        for m in recent
        if m.get("content")
    )


def condense_query(query: str, history: list[dict]) -> str:
    """
    Rewrite a follow-up into a question that stands on its own.

    Vector search sees only the text it is given, so "what about in Europe?"
    retrieves near-randomly: none of the words that identify the subject are in
    it. Condensing against the recent turns is what makes turn two onwards work.

    The opposite failure matters just as much — when the user genuinely changes
    subject, dragging the previous policy into the query sends the search to the
    wrong document — so the model is told to leave self-contained questions be.
    """
    if not history:
        return query

    prompt = f"""Given the conversation below, rewrite the user's latest message as a
standalone question that can be understood with no other context.

Rules:
- Resolve pronouns and references ("it", "that section", "what about X") using the conversation.
- Name the policy document or subject the question is about, when the conversation makes it clear.
- If the latest message is ALREADY self-contained, or changes to a new subject,
  return it unchanged.
- Return only the rewritten question. No preamble, no quotes.

Conversation:
{format_history(history)}

Latest message: {query}

Standalone question:"""

    try:
        rewritten = chat_model().invoke(prompt).content.strip()
    except Exception:
        return query        # a condensation failure must not lose the question
    return rewritten or query


# ---------------------------------------------------------------------------
# RAG answer
# ---------------------------------------------------------------------------

def answer_question(query: str, verbose: bool = False,
                    document_name: str | None = None,
                    history: list[dict] | None = None,
                    condense: bool = True) -> RagAnswer:
    """
    Full RAG chain: condense -> retrieve parent chunks -> LLM answer.

    `history` is a list of {"role", "content"} dicts, oldest first. Pass it to
    answer follow-up questions; it is used both to condense the search query and
    to give the model the conversation so far.

    `condense=False` searches on the raw message instead — useful for comparing
    retrieval with and without condensation.

    Returns a RagAnswer carrying the answer, the sections it came from, and the
    query actually searched on.
    """
    history = history or []
    search_query = condense_query(query, history) if (condense and history) else query

    if verbose and search_query != query:
        print(f"\n  Condensed: {query!r} -> {search_query!r}")

    parents = retrieve_parent_chunks(search_query, document_name)
    sources = [Source.from_parent(p) for p in parents]

    if not parents:
        return RagAnswer(
            answer="No relevant content found in the policy documents.",
            query=query,
            search_query=search_query,
            sufficient=False,
        )

    if verbose:
        print(f"\n  Retrieved {len(parents)} parent chunk(s):")
        for source in sources:
            print(f"    - {source}")

    ids = source_ids(parents)
    context = build_context(parents, ids)
    conversation = format_history(history)
    conversation_block = (
        f"\nConversation so far (for resolving what the question refers to):\n"
        f"{conversation}\n"
        if conversation else ""
    )

    prompt = f"""You are a helpful assistant answering questions about insurance policy documents.
Answer the question using ONLY the policy sections provided below.
If they do not answer it, say "I don't have enough information to answer that."
and set sufficient to false.

Each section is labelled with an id in square brackets, e.g. [S1]. Report the
ids you used in sources_used, most important first. The FIRST id must be the
section that most directly answers the question — the one defining the cover or
rule being asked about, not a section that merely adds a limit, an excess, an
exclusion or a summary of it. Prefer the policy wording that grants or defines
the cover over a summary document that restates it. Do not list sections you
only mention in passing, and never invent an id.

The sections may come from more than one policy document, grouped below under
"Policy document:" headings. When they do:
- Say which policy document each fact comes from.
- If the documents give different answers, present them separately. Never merge
  rules from different policies into a single figure or procedure.

Be concise. Do not repeat the section ids in the answer text itself.
{conversation_block}
Sections:
{context}

Question: {query}"""

    result = generate(prompt)
    return RagAnswer(
        answer=result.answer,
        query=query,
        search_query=search_query,
        sufficient=result.sufficient,
        sources=apply_citations(result, ids, sources),
    )


# ---------------------------------------------------------------------------
# Main — sample questions
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    questions = [
        "What is covered under Section A Roadside?",
        "What does 'beyond economical repair' mean?",
        "What happens if my vehicle breaks down on a French motorway?",
        "How do I make a complaint?",
        "Can I cancel my RAC Breakdown Cover?",
        "What alternative transport is available under Section D?",
    ]

    print("=" * 60)
    print("  RAG Retriever -- RAC Breakdown Policy")
    print("=" * 60)

    for q in questions:
        print(f"\nQ: {q}")
        result = answer_question(q, verbose=True)
        print(f"A: {result.answer}")
        for i, source in enumerate(result.sources, 1):
            print(f"   [{i}] {source.section or '(no heading)'}")
            print(f"       {source.parent_path}")
        print("-" * 60)
