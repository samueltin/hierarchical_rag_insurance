"""
rag_retriever.py

RAG query engine using the parent-child index built by doc_intell.py.

Query time flow:
  1. Embed user query via Azure OpenAI
  2. Vector search Azure AI Search  →  top matching child chunks
  3. Extract parent_ids from results (deduplicated)
  4. Fetch parent JSON blobs from Azure Blob Storage
  5. Build context from parent content + section metadata
  6. Pass context + question to LLM and return answer
"""

import os
import json
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
EMBEDDING_DEPLOYMENT     = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
CHAT_DEPLOYMENT          = os.getenv("CHAT_DEPLOYMENT", "gpt-4o")

BLOB_CONNECTION_STRING   = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME      = os.getenv("BLOB_CONTAINER_NAME", "parent-chunks")

SEARCH_ENDPOINT          = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX_NAME        = os.getenv("AZURE_SEARCH_INDEX_NAME", "breakdown-child-chunks")

TOP_K = 5  # number of child chunks to retrieve per query

# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve_parent_chunks(query: str) -> list[dict]:
    """
    Embed query -> vector search child chunks -> fetch parent blobs.
    Returns a list of parent dicts: {parent_id, content, metadata}.
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
        select=["child_id", "parent_id", "content",
                "meta_h1", "meta_h2", "meta_h3", "meta_h4"],
        top=TOP_K,
    ))

    # 3. Deduplicate parent_ids — multiple children may share a parent
    seen = set()
    parent_ids = []
    for r in results:
        pid = r["parent_id"]
        if pid not in seen:
            seen.add(pid)
            parent_ids.append(pid)

    # 4. Fetch parent blobs from Azure Blob Storage
    blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service.get_container_client(BLOB_CONTAINER_NAME)

    parents = []
    for parent_id in parent_ids:
        blob_client = container_client.get_blob_client(f"{parent_id}.json")
        data = blob_client.download_blob().readall()
        parents.append(json.loads(data))

    return parents


def build_context(parents: list[dict]) -> str:
    """Format parent chunks into a readable context block for the LLM."""
    sections = []
    for p in parents:
        meta = p.get("metadata", {})
        # Build a breadcrumb from available heading levels
        breadcrumb = " > ".join(
            v for v in [
                meta.get("H1", ""),
                meta.get("H2", ""),
                meta.get("H3", ""),
                meta.get("H4", ""),
            ] if v
        )
        header = f"[{breadcrumb}]" if breadcrumb else "[General]"
        sections.append(f"{header}\n{p['content']}")
    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# RAG answer
# ---------------------------------------------------------------------------

def answer_question(query: str, verbose: bool = False) -> str:
    """Full RAG chain: retrieve parent chunks -> LLM answer."""

    parents = retrieve_parent_chunks(query)

    if not parents:
        return "No relevant content found in the policy document."

    if verbose:
        print(f"\n  Retrieved {len(parents)} parent chunk(s):")
        for p in parents:
            meta = p.get("metadata", {})
            breadcrumb = " > ".join(v for v in [
                meta.get("H1", ""), meta.get("H2", ""),
                meta.get("H3", ""), meta.get("H4", ""),
            ] if v)
            print(f"    - {breadcrumb or '(no heading)'} "
                  f"[{len(p['content'])} chars]")

    context = build_context(parents)

    llm = AzureChatOpenAI(
        azure_deployment=CHAT_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=lambda: DefaultAzureCredential()
            .get_token("https://cognitiveservices.azure.com/.default").token,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
    )

    prompt = f"""You are a helpful assistant for RAC Breakdown Cover policy questions.
Answer the question using ONLY the policy context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."
Be concise and cite the section name where relevant.

Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return response.content


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
        answer = answer_question(q, verbose=True)
        print(f"A: {answer}")
        print("-" * 60)
