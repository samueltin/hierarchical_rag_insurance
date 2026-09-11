"""
Stage 4: EmbeddingEncoder — embed the child chunks and index them for search.

Context in:
    chunks_path        prefix URL holding one JSON blob per child chunk, e.g.
                       https://<account>.blob.core.windows.net/ingestion-pipeline/chunks/breakdown_policy_booklet/
    document_name      (optional) logical document name; derived from chunks_path if absent

Context out:
    index_name             search index the chunks were written to
    indexed_count          number of child chunks embedded and indexed
    embedding_model        deployment/model used to embed them
    embedding_dimensions   vector width of that model

The index holds chunks from many documents, so every search document carries
`document_name` as filterable metadata. Re-indexing a document only touches
that document's rows.
"""

from abc import ABC, abstractmethod
import os

from azure.core.exceptions import ResourceNotFoundError

from storage import download_json, list_blobs, parse_blob_url
from ..handler import PipelineContext, PipelineHandler

# Fields every search document carries, copied straight off the child chunk blob.
CHUNK_FIELDS = (
    "child_id",
    "parent_id",
    "parent_path",
    "document_name",
    "content",
    "meta_h1",
    "meta_h2",
    "meta_h3",
    "meta_h4",
)


class EmbeddingEncoder(PipelineHandler, ABC):
    """
    Base class for embedding encoders.

    Reads every child chunk under `chunks_path`, embeds the chunk text in
    batches, and upserts the result into a search index. Subclasses supply the
    embedding model and the index plumbing.
    """

    def __init__(
        self,
        index_name: str | None = None,
        embed_batch_size: int = 64,
        upload_batch_size: int = 100,
        prune: bool = True,
    ) -> None:
        """
        index_name:        target search index (default: AZURE_SEARCH_INDEX_NAME)
        embed_batch_size:  chunks per embedding request
        upload_batch_size: documents per index upload request
        prune:             delete index rows for chunks this document no longer has
        """
        super().__init__()
        self.index_name = index_name or os.getenv(
            "AZURE_SEARCH_INDEX_NAME", "breakdown-child-chunks"
        )
        self.embed_batch_size = embed_batch_size
        self.upload_batch_size = upload_batch_size
        self.prune = prune

    # -- PipelineHandler ----------------------------------------------------

    def process(self, context: PipelineContext) -> PipelineContext:
        chunks_dir = parse_blob_url(self.require(context, "chunks_path"))
        blobs = list_blobs(chunks_dir)
        if not blobs:
            raise ValueError(f"No chunks found under {chunks_dir.directory_url}")

        chunks = [download_json(blob) for blob in blobs]
        document_name = (
            context.get("document_name")
            or chunks[0].get("document_name")
            or chunks_dir.name.rstrip("/").rsplit("/", 1)[-1]
        )
        self.log(f"Embedding {len(chunks)} chunks from {chunks_dir.directory_url}")

        self.ensure_index()

        vectors = self._embed_all([chunk["content"] for chunk in chunks])
        documents = [
            {
                **{field: chunk.get(field, "") for field in CHUNK_FIELDS},
                "document_name": document_name,
                "content_vector": vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ]

        uploaded = self._upload_all(documents)
        if self.prune:
            self._prune_stale(document_name, {doc["child_id"] for doc in documents})

        context["index_name"] = self.index_name
        context["indexed_count"] = uploaded
        context["embedding_model"] = self.embedding_model
        context["embedding_dimensions"] = self.embedding_dimensions
        return context

    # -- subclass contract --------------------------------------------------

    @property
    @abstractmethod
    def embedding_model(self) -> str:
        """Deployment or model name used to embed — recorded so queries can match it."""

    @property
    @abstractmethod
    def embedding_dimensions(self) -> int:
        """Vector width the index must be built for."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts."""

    @abstractmethod
    def ensure_index(self) -> None:
        """Create the index if it does not exist; add any missing fields if it does."""

    @abstractmethod
    def upload(self, documents: list[dict]) -> int:
        """Upsert a batch of search documents. Returns how many were accepted."""

    @abstractmethod
    def existing_chunk_ids(self, document_name: str) -> set[str]:
        """Keys already indexed for this document."""

    @abstractmethod
    def delete(self, chunk_ids: set[str]) -> int:
        """Remove indexed documents by key."""

    # -- orchestration ------------------------------------------------------

    def _embed_all(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.embed_batch_size):
            batch = texts[start : start + self.embed_batch_size]
            vectors.extend(self.embed_documents(batch))
            self.log(f"Embedded {len(vectors)}/{len(texts)}")
        return vectors

    def _upload_all(self, documents: list[dict]) -> int:
        uploaded = 0
        for start in range(0, len(documents), self.upload_batch_size):
            batch = documents[start : start + self.upload_batch_size]
            uploaded += self.upload(batch)
        self.log(f"Indexed {uploaded} chunks into '{self.index_name}'")
        return uploaded

    def _prune_stale(self, document_name: str, current_ids: set[str]) -> None:
        """
        Drop rows for chunks that no longer exist.

        Chunk ids are deterministic, so re-indexing overwrites in place — but a
        changed chunk size produces fewer chunks, and the previous run's tail
        would otherwise keep matching queries forever.
        """
        stale = self.existing_chunk_ids(document_name) - current_ids
        if stale:
            self.log(f"Pruning {len(stale)} stale rows for '{document_name}'")
            self.delete(stale)


class AzureOpenAIEncoder(EmbeddingEncoder):
    """Embed with Azure OpenAI and index into Azure AI Search."""

    VECTOR_PROFILE = "hnsw-profile"
    VECTOR_ALGORITHM = "hnsw-algo"

    def __init__(
        self,
        index_name: str | None = None,
        embed_batch_size: int = 64,
        upload_batch_size: int = 100,
        prune: bool = True,
        deployment: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        super().__init__(index_name, embed_batch_size, upload_batch_size, prune)
        self.deployment = deployment or os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.dimensions = dimensions or int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self._embedder = None
        self._search_client = None

    # -- embedding ----------------------------------------------------------

    @property
    def embedding_model(self) -> str:
        return self.deployment

    @property
    def embedding_dimensions(self) -> int:
        return self.dimensions

    @property
    def embedder(self):
        if self._embedder is None:
            from langchain_openai import AzureOpenAIEmbeddings

            self._embedder = AzureOpenAIEmbeddings(
                azure_deployment=self.deployment,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_ad_token_provider=lambda: _credential()
                .get_token("https://cognitiveservices.azure.com/.default")
                .token,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
        return self._embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # One request per batch rather than one per chunk.
        return self.embedder.embed_documents(texts)

    # -- index --------------------------------------------------------------

    def ensure_index(self) -> None:
        from azure.search.documents.indexes import SearchIndexClient

        index_client = SearchIndexClient(
            endpoint=self.search_endpoint, credential=_credential()
        )
        try:
            existing = index_client.get_index(self.index_name)
        except ResourceNotFoundError:
            self.log(f"Creating search index '{self.index_name}'")
            index_client.create_index(self._index_definition())
            return

        # Index already exists — only add fields it is missing. Azure AI Search
        # allows new fields to be added, but not existing ones to be altered.
        present = {field.name for field in existing.fields}
        missing = [f for f in self._fields() if f.name not in present]
        if not missing:
            self.log(f"Search index '{self.index_name}' is up to date")
            return
        self.log(f"Adding {len(missing)} field(s) to '{self.index_name}': "
                 f"{', '.join(f.name for f in missing)}")
        existing.fields.extend(missing)
        index_client.create_or_update_index(existing)

    def _fields(self) -> list:
        from azure.search.documents.indexes.models import (
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SimpleField,
        )

        return [
            SimpleField(name="child_id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
            # Which document a chunk came from — the index spans many documents.
            SimpleField(
                name="document_name",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
                sortable=True,
            ),
            # Full blob URL of the parent chunk, so query time need not know the layout.
            SimpleField(name="parent_path", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            # Parent heading metadata — filterable for scoped RAG queries
            SimpleField(name="meta_h1", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="meta_h2", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="meta_h3", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="meta_h4", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.dimensions,
                vector_search_profile_name=self.VECTOR_PROFILE,
            ),
        ]

    def _index_definition(self):
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchIndex,
            VectorSearch,
            VectorSearchProfile,
        )

        return SearchIndex(
            name=self.index_name,
            fields=self._fields(),
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name=self.VECTOR_ALGORITHM)],
                profiles=[
                    VectorSearchProfile(
                        name=self.VECTOR_PROFILE,
                        algorithm_configuration_name=self.VECTOR_ALGORITHM,
                    )
                ],
            ),
        )

    # -- documents ----------------------------------------------------------

    @property
    def search_client(self):
        if self._search_client is None:
            from azure.search.documents import SearchClient

            self._search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.index_name,
                credential=_credential(),
            )
        return self._search_client

    def upload(self, documents: list[dict]) -> int:
        results = self.search_client.upload_documents(documents=documents)
        failed = [r for r in results if not r.succeeded]
        if failed:
            raise RuntimeError(
                f"{len(failed)} document(s) rejected by '{self.index_name}', "
                f"first error: {failed[0].error_message}"
            )
        return len(results)

    def existing_chunk_ids(self, document_name: str) -> set[str]:
        results = self.search_client.search(
            search_text="*",
            filter=f"document_name eq '{document_name}'",
            select=["child_id"],
        )
        return {r["child_id"] for r in results}

    def delete(self, chunk_ids: set[str]) -> int:
        self.search_client.delete_documents(
            documents=[{"child_id": cid} for cid in chunk_ids]
        )
        return len(chunk_ids)


_shared_credential = None


def _credential():
    """One DefaultAzureCredential shared by the embedder and both search clients."""
    global _shared_credential
    if _shared_credential is None:
        from azure.identity import DefaultAzureCredential

        _shared_credential = DefaultAzureCredential()
    return _shared_credential
