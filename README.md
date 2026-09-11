# hierarchical_rag_insurance

A Retrieval-Augmented Generation (RAG) system for complex insurance policy documents, built on Azure.

It combines a **hierarchical parent-child chunking strategy** with a **configuration-driven ingestion pipeline**, a **document admin console**, and a **multi-turn chat interface** with source citations and groundedness checking.

---

## The Problem with Flat Chunking

Standard RAG splits a document into fixed-size chunks and indexes them directly. For complex insurance documents this creates two problems:

- **Small chunks** are precise enough for vector search but too short to give the LLM sufficient context to answer a question fully.
- **Large chunks** give the LLM enough context but produce diluted embeddings that reduce search accuracy.

---

## Solution: Hierarchical Parent-Child Chunking

Two levels of chunks with different purposes:

| Level | Size | Purpose | Storage |
|---|---|---|---|
| **Parent chunk** | Full document section | Sent to the LLM as context | Azure Blob Storage |
| **Child chunk** | ~400 characters | Embedded and searched | Azure AI Search |

At query time, vector search runs on **child chunks** for precision. Each child carries the blob URL of its parent, so the matched parent is fetched and passed to the LLM for a complete answer.

This also makes the system resilient to tables being split. A fees table spread across three 400-character children still reaches the LLM intact, because the parent holds the whole table.

---

## Architecture

### Ingestion — a Chain of Responsibility

Four stages, each with its own abstract base class, passing a shared context dictionary:

```
DocCracker  →  PreProcessor  →  DocChunker  →  EmbeddingEncoder
    │               │               │                │
 PDF → MD      clean MD       parent+child      vectors → index
```

| Stage | Base class | Implementations |
|---|---|---|
| 1. Crack | `DocCracker` | `PDFCracker` (Azure Document Intelligence) |
| 2. Pre-process | `PreProcessor` | `MarkdownCleaner`, `CarInsuranceMarkdownCleaner` |
| 3. Chunk | `DocChunker` | `ParentChildDocChunker` |
| 4. Encode | `EmbeddingEncoder` | `AzureOpenAIEncoder` |

Each stage writes its output to blob storage and records the URL in the context, so any stage can be re-run on its own without repeating the expensive ones.

Which implementation runs for a given document is **stored per document**, not hardcoded — see [Configuration-driven pipelines](#3-configuration-driven-pipelines).

### Query

```
User query
     │
     ▼
Query condensation          (multi-turn only: rewrite a follow-up as a standalone question)
     │
     ▼
Azure OpenAI Embeddings     (embed the condensed query)
     │
     ▼
Azure AI Search             (hybrid search: vector + keyword, over child chunks)
     │
     ▼
Parent ranking              (group children by parent, sum their scores)
     │
     ▼
Azure Blob Storage          (fetch parent chunks by parent_path)
     │
     ▼
Azure OpenAI Chat           (answer, grouped and attributed by policy document)
     │
     ▼
Content Safety              (groundedness check, advisory)
     │
     ▼
Answer + cited sources
```

### Services

| Service | Port | Purpose |
|---|---|---|
| Chat API | 8000 | Question answering, conversation history |
| Admin API | 8100 | Document registry, upload, pipeline triggering |
| Chat UI | 8501 | Multi-turn chat with citations |
| Admin console | 8503 | Manage documents and their pipeline recipes |

---

## Key Design Decisions

### 1. Azure Document Intelligence over PyMuPDF

Insurance policy PDFs are often scanned or image-based. PyMuPDF with Tesseract OCR loses all heading structure — every section title becomes bold body text. Azure Document Intelligence `prebuilt-layout` outputs structured Markdown with a proper `#` / `##` / `###` hierarchy.

The cracker also batches pages: a single request commonly returns only the first two pages of a long booklet, so it detects short reads and re-analyses in page batches.

### 2. Markdown Header Splitting for Parent Chunks

`MarkdownHeaderTextSplitter` respects the document's own logical structure rather than splitting on arbitrary character counts. Each parent chunk maps to a meaningful section and carries heading metadata usable for filtered search.

### 3. Configuration-driven Pipelines

Every registered document has a row in an Azure Table recording which implementation to use at each stage:

```
filename           car-insurance-policy-booklet.pdf
doc_cracker        PDFCracker
preprocessor       CarInsuranceMarkdownCleaner
doc_chunker        ParentChildDocChunker
embedding_encoder  AzureOpenAIEncoder
```

The row *is* the pipeline recipe. This matters because cleaning rules are document-specific — the breakdown booklet's back cover and the car insurance booklet's differ, and applying the wrong cleaner leaves legal boilerplate in the index or strips real content.

Stage names are resolved through an explicit registry built by introspecting concrete subclasses of each base class (`pipeline/registry.py`). Adding a new `PreProcessor` subclass makes it available in the admin console automatically. Resolution is deliberately **not** a dotted-path import: the table is admin-writable, and importing an arbitrary path from it would turn a metadata field into code execution.

### 4. Breadcrumb Enrichment on Child Embeddings

Sub-section content often does not mention its own section name, so a query naming the section would miss it. The heading breadcrumb is prepended to each child chunk **before embedding**:

```
2. Alternative transport > Covered
If the driver would prefer to continue the journey by air, rail,
taxi or public transport, the RAC will reimburse you for a standard
class ticket up to £150 per person or £500 for the whole party,
whichever is less.
```

The breadcrumb goes into the **embedded child only**. The parent chunk kept for LLM context remains clean policy language.

### 5. Deterministic Chunk IDs

Chunk IDs are a hash of document name, breadcrumb and ordinal rather than a UUID:

```
parent_id = sha256(document_name | breadcrumb | ordinal)[:32]
child_id  = sha256(parent_id | ordinal)[:32]
```

Re-ingesting a document therefore overwrites its chunks in place, in both blob storage and the search index. With random IDs, every re-run would add a second copy of every chunk to the index while the originals stayed and kept matching queries. Chunks left behind by a changed chunk size are pruned explicitly.

### 6. Self-contained Child Chunks

Each child blob carries its `parent_path` (a full blob URL) and a denormalised copy of the parent's heading metadata. The encoder can therefore build a search document from a child alone, and query time never needs to know how the pipeline lays out storage.

### 7. Parent Ranking by Summed Child Score

Ranking parents by their single best-matching child favours short chunks: a 108-character cross-reference that is almost entirely query terms out-scores the 420-character section that answers the question, on both keyword and vector search. Parents are ranked by the **sum** of their matching children's scores, so a section matching repeatedly outranks a short incidental match.

### 8. Cross-document Attribution

The corpus holds several policies that share section names — each has its own "Cancellation rights", "Complaints procedure" and "General exclusions". Retrieved sections are grouped by policy document in the prompt and labelled `[document › breadcrumb]`, and the model is instructed to attribute each fact and to present differing policies separately rather than merging them into one figure.

### 9. Query Condensation for Multi-turn Chat

Vector search sees only the text it is given, so a follow-up like *"What about in Europe?"* retrieves near-randomly. Before searching, a follow-up is rewritten into a standalone question using the recent turns. The condensed query is stored on the message, so a poor answer can be traced to the condenser or to the search rather than guessed at.

The condenser is also instructed to leave self-contained messages unchanged, so a genuine change of subject does not drag the previous policy along.

### 10. Groundedness Checking

Answers are checked against the retrieved sections using Azure AI Content Safety groundedness detection, and the verdict is shown in the chat UI. The check is **advisory**: if it is unconfigured, errors or times out, the answer is still returned with a note that it could not be verified.

By default the answer is checked against **all** retrieved sections, since good answers routinely combine several. `GROUNDEDNESS_SOURCES=best` checks against the top section only, which is stricter and will flag legitimate answers.

> **Region matters.** Groundedness detection is not available on every Content Safety resource. An unsupported resource returns HTTP 200 with an empty verdict, so every answer reads as 100% grounded. Verify with a known-ungrounded example before trusting the badge.

### 11. Hybrid Search

The retriever uses vector and keyword search together (`search_text` alongside `vector_queries`). Azure AI Search's Reciprocal Rank Fusion merges the two result sets, which helps for section-specific queries where exact terms ("Section A", "Complaints") are strong signals.

### 12. Authentication

Azure OpenAI, Azure AI Search and Azure Document Intelligence use `DefaultAzureCredential`, supporting managed identity in production and developer credentials locally. Blob and Table Storage use the account connection string, and Content Safety uses a resource key.

> Running in a container requires a managed identity or service principal environment variables — there is no Azure CLI inside the image for `DefaultAzureCredential` to fall back on.

---

## Project Structure

```
hierarchical_rag_insurance/
├── pipeline/                  # Ingestion — knows nothing about the admin table
│   ├── handler.py             #   PipelineHandler: chain wiring, shared context
│   ├── registry.py            #   stage name → class, by ABC introspection
│   ├── runner.py              #   recipe + source URL → runs the chain
│   └── stages/
│       ├── doc_cracker.py     #   DocCracker, PDFCracker
│       ├── preprocessor.py    #   PreProcessor, MarkdownCleaner, CarInsuranceMarkdownCleaner
│       ├── doc_chunker.py     #   DocChunker, ParentChildDocChunker
│       └── embedding_encoder.py #  EmbeddingEncoder, AzureOpenAIEncoder
├── storage/                   # Azure Blob access, shared by ingestion and chat
│   └── blob_storage.py        #   BlobLocation, upload/download/list helpers
├── admin/                     # Document admin console
│   ├── api.py                 #   FastAPI: list, upload, edit, trigger ingestion
│   ├── repository.py          #   Azure Table Storage — the only module that knows
│   ├── runner.py              #   row → run pipeline → write status back (+ CLI)
│   └── ui.py                  #   Streamlit console
├── chat/                      # Chat service support
│   ├── history.py             #   conversation transcripts in blob storage
│   └── groundedness.py        #   Content Safety groundedness checking
├── api/main.py                # Chat API: /ask, conversations, /chunk
├── ui/app.py                  # Chat UI
├── rag_retriever.py           # Retrieval, condensation, answer generation
├── ingest_pipeline.py         # Superseded monolith, kept for reference — not imported
├── docs/                      # Source PDFs
├── requirements.txt
├── env.example                # Copy to .env
└── .env                       # Azure credentials (not committed)
```

---

## Storage Layout

**Blob container `ingestion-pipeline`** — one prefix per pipeline stage:

```
source-documents/<filename>.pdf
extracted-markdowns/<document>.md
clean-markdowns/<document>_clean.md
parent-chunks/<document>/<parent_id>.json
chunks/<document>/<child_id>.json
```

**Blob container `chat-history`** — one JSON blob per conversation:

```
<user>/<conversation_id>.json
```

Titles and timestamps live in blob metadata, so listing a user's conversations costs one call rather than one download per conversation. Transcripts store citations as references (section, document, `parent_path`) rather than chunk text, keeping them small; the chat UI fetches the wording on demand via `GET /chunk`.

**Azure Table `documents`** — one row per registered document: `PartitionKey = doc_type`, `RowKey = id`, plus the pipeline recipe, ingestion status and last-run statistics.

**Azure AI Search index** — child chunks with `document_name` as a filterable, facetable field, so one index serves multiple policy documents and a chat can be scoped to one or search across all.

---

## Prerequisites

- Python 3.11+
- Azure subscription with:
  - Azure Document Intelligence
  - Azure OpenAI (an embedding deployment and a chat deployment)
  - Azure Blob Storage **and** Table Storage (same account)
  - Azure AI Search
  - Azure AI Content Safety *(optional — groundedness checking; must be in a region that supports it)*

---

## Installation

```bash
git clone https://github.com/samueltin/hierarchical_rag_insurance.git
cd hierarchical_rag_insurance
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

Copy `env.example` to `.env` and fill in your Azure resource details:

```bash
# Azure Document Intelligence
DOC_INTEL_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
EMBEDDING_DEPLOYMENT=text-embedding-3-small
CHAT_DEPLOYMENT=gpt-4.1

# Azure Storage — the account is taken from AccountName here
BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
DOCUMENT_TABLE_NAME=documents

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://<your-search>.search.windows.net
AZURE_SEARCH_INDEX_NAME=policy-child-chunks

# Azure AI Content Safety (optional)
AZURE_CONTENT_SAFETY_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_CONTENT_SAFETY_KEY=
GROUNDEDNESS_SOURCES=all
GROUNDEDNESS_ENABLED=true
```

Optional settings with sensible defaults: `INGESTION_CONTAINER`, `SOURCE_DOCUMENT_PREFIX`, `EXTRACTED_MARKDOWN_PREFIX`, `CLEAN_MARKDOWN_PREFIX`, `CHUNK_PREFIX`, `PARENT_CHUNK_PREFIX`, `CHAT_HISTORY_CONTAINER`, `CHAT_USERS`, `CHAT_HISTORY_TURNS`, `MAX_UPLOAD_MB`, `DOC_INTEL_PAGE_BATCH`.

The storage account name is **never** hardcoded — it is derived from `BLOB_CONNECTION_STRING`, or set explicitly with `BLOB_ACCOUNT_URL` / `STORAGE_ACCOUNT_NAME`.

---

## Running

Each service runs from the repository root:

```bash
# Chat API + UI
.venv/bin/uvicorn api.main:app --reload --port 8000
.venv/bin/streamlit run ui/app.py --server.port 8501

# Admin API + console
.venv/bin/uvicorn admin.api:app --reload --port 8100
.venv/bin/streamlit run admin/ui.py --server.port 8503
```

---

## Usage

### Ingesting a document

Through the admin console at `http://localhost:8503`:

1. **Upload** a PDF, choosing the implementation for each pipeline stage.
2. **Run ingestion** — the API accepts the request and runs the pipeline in the background.
3. Watch `status` move `uploaded → running → succeeded`, with `stage_reached` showing progress. A failure records the stage and the error on the row.

Or from the command line:

```bash
python -m admin.runner --list           # registered documents and their status
python -m admin.runner <document-id>    # ingest one document
python -m admin.runner --all            # ingest everything not yet succeeded
```

Re-ingesting a document whose file has not changed skips the Document Intelligence call, which is the expensive stage — the content hash tells the runner whether the extraction is still valid.

### Asking questions

Through the chat UI at `http://localhost:8501`: pick a user, start a chat, and ask. Each answer shows its top matching section beside it, with the remaining retrieved sections behind an expander and a groundedness verdict beneath.

Or programmatically:

```python
from rag_retriever import answer_question

result = answer_question("What is covered under Section A Roadside?")
print(result.answer)
for source in result.sources:
    print(source.document_name, "›", source.section)
```

`answer_question` returns a `RagAnswer` carrying the answer, the sources, and the query that retrieval actually ran on. Pass `history=[{"role": ..., "content": ...}]` for follow-up questions, and `document_name=` to scope the search to one document.

---

## Domain Context

This implementation uses Aviva Zero motor and RAC breakdown policy documents — representative examples of insurance policy documents, which share characteristics that make RAG challenging:

- Scanned or image-based PDFs with no embedded text structure
- Deep heading hierarchies (policy → section → sub-section → clause)
- Definition sections where term names and definitions must stay linked
- Cross-references between sections, which are short and match queries strongly while carrying no information
- Generic sub-headings ("Covered", "Not Covered") repeated across sections
- The same concepts — cancellation, complaints, exclusions — appearing in several policies with different terms

The techniques here apply to any complex structured document in regulated industries: insurance policies, legal contracts, compliance frameworks, clinical guidelines.

---

## Author

Samuel Tin — Senior AI Architect
[LinkedIn](https://www.linkedin.com/in/samueltin) · [GitHub](https://github.com/samueltin)
