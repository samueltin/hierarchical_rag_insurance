# hierarchical_rag_insurance

A production-grade Retrieval-Augmented Generation (RAG) pipeline for complex insurance policy documents, built on Azure. It uses a **hierarchical parent-child chunking strategy** to improve retrieval precision and deliver richer context to the LLM — going beyond the flat chunking approach used in basic RAG implementations.

---

## The Problem with Flat Chunking

Standard RAG splits a document into fixed-size chunks and indexes them directly. For complex insurance documents this creates two problems:

- **Small chunks** are precise enough for vector search but too short to give the LLM sufficient context to answer a question fully.
- **Large chunks** give the LLM enough context but produce diluted embeddings that reduce search accuracy.

---

## Solution: Hierarchical Parent-Child Chunking

This pipeline solves both problems by maintaining two levels of chunks with different purposes:

| Level | Size | Purpose | Storage |
|---|---|---|---|
| **Parent chunk** | Full document section | Sent to LLM as context | Azure Blob Storage |
| **Child chunk** | ~400 tokens | Embedded and searched | Azure AI Search |

At query time, vector search runs on **child chunks** for precision. The matched child's `parent_id` is then used to fetch the **parent chunk** from Blob Storage, which is passed to the LLM for a complete, accurate answer.

---

## Architecture

```
PDF Document
     │
     ▼
Azure Document Intelligence          (prebuilt-layout → Markdown)
     │
     ▼
Markdown Cleaning                    (remove noise, normalise headings)
     │
     ▼
MarkdownHeaderTextSplitter           (split by H1 / H2 / H3 / H4 → parent chunks)
     │
     ├──────────────────────────────────────────────────┐
     ▼                                                  ▼
RecursiveCharacterTextSplitter       Azure Blob Storage
(parent → child chunks)              (one JSON blob per parent chunk)
     │
     ▼
Breadcrumb Enrichment                (prepend section path to child content)
     │
     ▼
Azure OpenAI Embeddings              (text-embedding-3-small)
     │
     ▼
Azure AI Search Index                (child chunks + vectors + parent metadata)
```

**At query time:**

```
User query
     │
     ▼
Azure OpenAI Embeddings              (embed the query)
     │
     ▼
Azure AI Search                      (hybrid search: vector + keyword)
     │  returns top-K child chunks with parent_id
     ▼
Azure Blob Storage                   (fetch parent chunk by parent_id)
     │
     ▼
Azure OpenAI Chat                    (LLM generates answer from parent context)
     │
     ▼
Answer
```

---

## Key Design Decisions

### 1. Azure Document Intelligence over PyMuPDF

Insurance policy PDFs are often scanned or image-based. PyMuPDF with Tesseract OCR loses all heading structure — every section title becomes bold body text. Azure Document Intelligence `prebuilt-layout` uses a layout model trained on real documents and outputs structured Markdown with proper `#` / `##` / `###` heading hierarchy.

### 2. Markdown Header Splitting for Parent Chunks

Using `MarkdownHeaderTextSplitter` respects the document's own logical structure rather than splitting on arbitrary character counts. Each parent chunk maps to a meaningful section (e.g. "Section A. Roadside", "Definition of words > Beyond economical repair") and carries heading metadata that can be used for filtered search.

### 3. Breadcrumb Enrichment on Child Embeddings

A key limitation of heading-based splitting is that sub-section content often does not mention its own section name. For example, a chunk about alternative transport reimbursement (£150/person) contains no reference to "Section D" in its text. Without enrichment, a query about "Section D alternative transport" would miss it.

The fix is to prepend the heading breadcrumb to each child chunk's content **before embedding**:

```
Section D. Onward Travel > 2. Alternative transport > Covered
If the driver would prefer to continue the journey by air, rail,
taxi or public transport, the RAC will reimburse you for a standard
class ticket up to £150 per person or £500 for the whole party.
```

The breadcrumb is added to the **embedded content only**. The parent chunk stored in Blob Storage remains clean policy language — the LLM never sees the breadcrumb.

### 4. Keyless Authentication

All Azure service clients use `DefaultAzureCredential` rather than API keys. This supports managed identity in production and developer credentials (Azure CLI / VS Code) in local development without any code changes.

### 5. Hybrid Search

The retriever uses both vector search and keyword search together (`search_text=query` alongside `vector_queries`). Azure AI Search's Reciprocal Rank Fusion (RRF) merges the two result sets. This improves precision for section-specific queries where exact terms (e.g. "Section A", "Complaints") are strong signals.

---

## Project Structure

```
hierarchical_rag_insurance/
├── docs/
│   └── breakdown_policy_booklet.pdf   # Source document
│   └── output.md                          # Raw Markdown from Azure Document Intelligence
│   └── output_clean.md                    # Cleaned Markdown (generated, not committed)
├── ingest_pipeline.py                 # Ingestion: extract → clean → chunk → index
├── rag_retriever.py                   # Query: search → retrieve parent → LLM answer
├── requirements.txt
└── .env                               # Azure credentials (not committed)
```

---

## Prerequisites

- Python 3.11+
- Azure subscription with the following services provisioned:
  - Azure Document Intelligence
  - Azure OpenAI (with `text-embedding-3-small` and a chat model deployed)
  - Azure Blob Storage
  - Azure AI Search

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

Copy `.env.example` to `.env` and fill in your Azure resource details:

```bash
# Azure Document Intelligence
DOC_INTEL_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
EMBEDDING_DEPLOYMENT=text-embedding-3-small
CHAT_DEPLOYMENT=gpt-4o

# Azure Blob Storage
BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
BLOB_CONTAINER_NAME=parent-chunks

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://<your-search>.search.windows.net
AZURE_SEARCH_INDEX_NAME=breakdown-child-chunks
```

---

## Usage

**Step 1 — Build the index** (run once, or whenever the source document changes):

```bash
python ingest_pipeline.py
```

This will:
- Extract structured Markdown from the PDF via Azure Document Intelligence
- Clean and normalise the Markdown
- Split into parent chunks (by heading structure) and child chunks (by size)
- Upload parent chunks to Azure Blob Storage
- Create the Azure AI Search index and upload embedded child chunks

**Step 2 — Query the index:**

```bash
python rag_retriever.py
```

To use the retriever in your own code:

```python
from rag_retriever import answer_question

answer = answer_question("What is covered under Section A Roadside?", verbose=True)
print(answer)
```

---

## Requirements

```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-openai>=0.2.0
langchain-text-splitters>=0.3.0
azure-ai-documentintelligence>=1.0.0
azure-search-documents>=11.6.0
azure-storage-blob>=12.22.0
azure-identity>=1.19.0
azure-core>=1.32.0
openai>=1.50.0
pymupdf4llm>=0.0.17
pymupdf>=1.24.0
python-dotenv>=1.0.0
```

---

## Domain Context

This implementation uses an RAC motor breakdown policy booklet as the source document — a representative example of insurance policy documents which share common characteristics that make RAG challenging:

- Scanned or image-based PDFs with no embedded text structure
- Deep heading hierarchies (policy → section → sub-section → clause)
- Definition sections where term names and their definitions must stay linked
- Cross-references between sections
- Generic sub-headings ("Covered", "Not Covered") repeated across multiple sections

The hierarchical chunking strategy and breadcrumb enrichment technique in this project are applicable to any complex structured document in regulated industries — insurance policies, legal contracts, compliance frameworks, and clinical guidelines.

---

## Author

Samuel Tin — Senior AI Architect  
[LinkedIn](https://www.linkedin.com/in/samueltin) · [GitHub](https://github.com/samueltin)
