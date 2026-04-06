"""
doc_intell.py

Full parent-child chunking pipeline using Azure Document Intelligence.

Pipeline steps:
  1. Extract Markdown from PDF via Azure Document Intelligence
  2. Clean the Markdown (remove noise, normalise headings)
  3. Split into parent chunks via MarkdownHeaderTextSplitter
  4. Split each parent into child chunks via RecursiveCharacterTextSplitter
  5. Upload parent chunks → Azure Blob Storage (one JSON blob per parent)
  6. Create Azure AI Search index (child chunks + vectors + parent metadata)
  7. Embed and upload child chunks → Azure AI Search

Query time flow (see rag_retriever.py):
  User query → embed → vector search (children) → fetch parent blob → LLM
"""

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from pathlib import Path
from dotenv import load_dotenv
import os
import re
import json
import uuid
import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

DOC_PATH = Path(__file__).parent / "docs/breakdown_policy_booklet.pdf"

# Azure Document Intelligence
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")

# Azure OpenAI
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
EMBEDDING_DEPLOYMENT     = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
EMBEDDING_DIMENSIONS     = 1536

# Azure Blob Storage
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME    = os.getenv("BLOB_CONTAINER_NAME", "parent-chunks")

# Azure AI Search
SEARCH_ENDPOINT    = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX_NAME  = os.getenv("AZURE_SEARCH_INDEX_NAME", "breakdown-child-chunks")

# Chunking
CHILD_CHUNK_SIZE    = 400
CHILD_CHUNK_OVERLAP = 50
MIN_CHUNK_CHARS     = 50   # discard chunks shorter than this

if not DOC_INTEL_ENDPOINT:
    raise ValueError("Missing DOC_INTEL_ENDPOINT in .env")

# ---------------------------------------------------------------------------
# Step 1: Extract Markdown via Azure Document Intelligence
# ---------------------------------------------------------------------------

def extract_markdown(doc_path: Path) -> str:
    print("\n[1/7] Extracting Markdown via Azure Document Intelligence...")

    client = DocumentIntelligenceClient(
        endpoint=DOC_INTEL_ENDPOINT,
        credential=DefaultAzureCredential(),
    )

    with fitz.open(doc_path) as pdf:
        page_count = pdf.page_count

    def analyze_pages(pages: str | None = None):
        with open(doc_path, "rb") as f:
            poller = client.begin_analyze_document(
                "prebuilt-layout",
                body=f,
                output_content_format="markdown",
                pages=pages,
            )
        return poller.result()

    result = analyze_pages()
    markdown_text = result.content

    if page_count > len(result.pages):
        batch_size = int(os.getenv("DOC_INTEL_PAGE_BATCH", "2"))
        all_md: list[str] = []
        for start in range(1, page_count + 1, batch_size):
            end = min(start + batch_size - 1, page_count)
            page_spec = f"{start}-{end}" if start != end else str(start)
            batch = analyze_pages(page_spec)
            all_md.append(batch.content)
        markdown_text = "\n\n".join(all_md)

    # Save raw output for inspection
    raw_path = doc_path.parent / "output.md"
    raw_path.write_text(markdown_text, encoding="utf-8")
    print(f"      Pages detected : {len(result.pages)}")
    print(f"      Total chars    : {len(markdown_text):,}")
    print(f"      Raw saved to   : {raw_path}")
    return markdown_text

# ---------------------------------------------------------------------------
# Step 2: Clean the Markdown
# ---------------------------------------------------------------------------

# Exact back cover paragraph text — use str.replace, NOT regex, to avoid
# spanning from the cover page occurrence to the back cover occurrence.
_BACK_COVER_PARA = (
    "If you need breakdown assistance ...\n"
    "call us straight away on 0345 305 7555.\n"
    "For our joint protection calls may be recorded and/or monitored."
)

def clean_markdown(md: str) -> str:
    """
    Fix 1: Remove HTML page marker comments (<!-- PageBreak -->, etc.)
    Fix 2: Remove Table of Contents (duplicates all section titles — pollutes search)
    Fix 3a: Remove back cover H1 "# Retirement | Investments ..." and everything after
    Fix 3b: Remove orphaned back cover paragraph via exact str.replace
            (regex with DOTALL would span from cover page to back cover, wiping the document)
    Fix 4: Remove logo OCR text before first heading via \\A anchor
            (^ with MULTILINE would match every line start, wiping content before every H1)
    Fix 5: Normalise #### → ### inside Definition of words section
            (Azure Doc Intelligence inconsistently assigned heading levels to definition terms)
    """
    md = re.sub(r'<!--.*?-->\n?', '', md)                                          # Fix 1
    md = re.sub(r'## Contents\n.*?(?=\n# )', '', md, flags=re.DOTALL)             # Fix 2
    md = re.sub(r'\n# Retirement\b.*', '', md, flags=re.DOTALL)                   # Fix 3a
    md = md.replace(_BACK_COVER_PARA, '')                                          # Fix 3b
    md = re.sub(r'\A.*?(?=^# )', '', md, flags=re.DOTALL | re.MULTILINE)          # Fix 4

    def _normalise(match):                                                          # Fix 5
        return match.group(0).replace('\n#### ', '\n### ')
    md = re.sub(r'(## Definition of words.*?)(?=\n## |\n# )', _normalise, md, flags=re.DOTALL)

    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip()

# ---------------------------------------------------------------------------
# Step 3: Split into parent chunks
# ---------------------------------------------------------------------------

def split_parent_chunks(markdown_clean: str) -> list:
    print("\n[3/7] Splitting into parent chunks...")

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",    "H1"),
            ("##",   "H2"),
            ("###",  "H3"),
            ("####", "H4"),
        ]
    )
    raw_chunks = splitter.split_text(markdown_clean)
    parent_chunks = [c for c in raw_chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]

    print(f"      Raw chunks      : {len(raw_chunks)}")
    print(f"      After filter    : {len(parent_chunks)}  "
          f"(removed {len(raw_chunks) - len(parent_chunks)} short chunks)")
    return parent_chunks

# ---------------------------------------------------------------------------
# Step 4: Build parent-child pairs
# ---------------------------------------------------------------------------

def build_parent_child_pairs(parent_chunks: list) -> list[dict]:
    """
    Returns a list of records:
    {
        "parent_id":       str (UUID),
        "parent_content":  str,
        "parent_metadata": dict  e.g. {"H1": "Your Cover", "H2": "Section A. Roadside"},
        "children": [
            {"child_id": str, "content": str},
            ...
        ]
    }
    """
    print("\n[4/7] Splitting each parent into child chunks...")

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    records = []
    total_children = 0

    for parent_doc in parent_chunks:
        parent_id = str(uuid.uuid4())
        meta = parent_doc.metadata

        # Build breadcrumb from heading metadata e.g. "Your Cover > Section A. Roadside"
        # Prepend it to each child chunk's content before embedding so that vector
        # search can match section-specific queries (e.g. "Section D alternative transport")
        # even when the chunk text itself does not contain the section name.
        # The breadcrumb is added to the embedded content only — parent_content
        # stored in Blob Storage remains the original clean text for LLM context.
        breadcrumb = " > ".join(
            v for v in [meta.get("H1",""), meta.get("H2",""),
                        meta.get("H3",""), meta.get("H4","")]
            if v
        )

        children_text = child_splitter.split_text(parent_doc.page_content)
        children = [
            {
                "child_id": str(uuid.uuid4()),
                "content":  f"{breadcrumb}\n{text}" if breadcrumb else text,
            }
            for text in children_text
        ]
        total_children += len(children)
        records.append({
            "parent_id":       parent_id,
            "parent_content":  parent_doc.page_content,
            "parent_metadata": parent_doc.metadata,
            "children":        children,
        })

    print(f"      Parent chunks   : {len(records)}")
    print(f"      Total children  : {total_children}")
    print(f"      Avg children    : {total_children / len(records):.1f} per parent")
    return records

# ---------------------------------------------------------------------------
# Step 5: Upload parent chunks to Azure Blob Storage
# ---------------------------------------------------------------------------

def upload_parents_to_blob(records: list[dict]) -> None:
    print(f"\n[5/7] Uploading {len(records)} parent chunks to Blob Storage "
          f"(container: '{BLOB_CONTAINER_NAME}')...")

    blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service.get_container_client(BLOB_CONTAINER_NAME)

    if not container_client.exists():
        container_client.create_container()
        print(f"      Created container: {BLOB_CONTAINER_NAME}")

    for record in records:
        payload = {
            "parent_id": record["parent_id"],
            "content":   record["parent_content"],
            "metadata":  record["parent_metadata"],
        }
        container_client.upload_blob(
            name=f"{record['parent_id']}.json",
            data=json.dumps(payload, ensure_ascii=False),
            overwrite=True,
        )

    print(f"      Done. {len(records)} blobs uploaded.")

# ---------------------------------------------------------------------------
# Step 6: Create Azure AI Search index
# ---------------------------------------------------------------------------

def create_search_index() -> None:
    print(f"\n[6/7] Creating / updating Azure AI Search index '{SEARCH_INDEX_NAME}'...")

    index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=DefaultAzureCredential(),
    )

    fields = [
        SimpleField(name="child_id",  type=SearchFieldDataType.String, key=True),
        SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        # Parent heading metadata — filterable for scoped RAG queries
        SimpleField(name="meta_h1", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="meta_h2", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="meta_h3", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="meta_h4", type=SearchFieldDataType.String, filterable=True),
        # Vector field for semantic search
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[VectorSearchProfile(
            name="hnsw-profile",
            algorithm_configuration_name="hnsw-algo",
        )],
    )

    index_client.create_or_update_index(
        SearchIndex(name=SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search)
    )
    print("      Index ready.")

# ---------------------------------------------------------------------------
# Step 7: Embed and upload child chunks to Azure AI Search
# ---------------------------------------------------------------------------

def embed_and_upload_children(records: list[dict]) -> None:
    total_children = sum(len(r["children"]) for r in records)
    print(f"\n[7/7] Embedding and uploading {total_children} child chunks to Azure AI Search...")

    embedder = AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=lambda: DefaultAzureCredential()
            .get_token("https://cognitiveservices.azure.com/.default").token,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=DefaultAzureCredential(),
    )

    BATCH_SIZE = 100
    batch = []
    uploaded = 0

    for record in records:
        meta = record["parent_metadata"]

        for child in record["children"]:
            embedding = embedder.embed_query(child["content"])

            batch.append({
                "child_id":       child["child_id"],
                "parent_id":      record["parent_id"],
                "content":        child["content"],
                "meta_h1":        meta.get("H1", ""),
                "meta_h2":        meta.get("H2", ""),
                "meta_h3":        meta.get("H3", ""),
                "meta_h4":        meta.get("H4", ""),
                "content_vector": embedding,
            })

            if len(batch) >= BATCH_SIZE:
                search_client.upload_documents(documents=batch)
                uploaded += len(batch)
                print(f"      Uploaded {uploaded}/{total_children}...")
                batch = []

    if batch:
        search_client.upload_documents(documents=batch)
        uploaded += len(batch)

    print(f"      Done. {uploaded} child chunks indexed.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Parent-Child Chunking Pipeline (Azure Document Intelligence)")
    print("=" * 60)

    # Step 1 & 2: Extract and clean
    markdown_raw   = extract_markdown(DOC_PATH)

    print("\n[2/7] Cleaning Markdown...")
    markdown_clean = clean_markdown(markdown_raw)
    clean_path = DOC_PATH.parent / "output_clean.md"
    clean_path.write_text(markdown_clean, encoding="utf-8")
    print(f"      Total chars     : {len(markdown_clean):,}")
    print(f"      Clean saved to  : {clean_path}")

    # Steps 3 & 4: Chunk
    parent_chunks = split_parent_chunks(markdown_clean)
    records       = build_parent_child_pairs(parent_chunks)

    # Steps 5–7: Store
    upload_parents_to_blob(records)
    create_search_index()
    embed_and_upload_children(records)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Parent chunks → Blob Storage  : {BLOB_CONTAINER_NAME}")
    print(f"  Child chunks  → AI Search     : {SEARCH_INDEX_NAME}")
    print("=" * 60)
