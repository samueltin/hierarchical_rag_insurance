"""
Stage 3: DocChunker — split the cleaned Markdown into chunks and persist them.

Context in:
    clean_markdown_path  full blob URL of the cleaned Markdown, e.g.
                         https://<account>.blob.core.windows.net/ingestion-pipeline/clean-markdowns/breakdown_policy_booklet_clean.md
    document_name        (optional) logical document name; derived from the URL if absent

Context out:
    chunks_path          prefix URL holding one JSON blob per child chunk, e.g.
                         https://<account>.blob.core.windows.net/ingestion-pipeline/chunks/breakdown_policy_booklet/
    parent_chunks_path   prefix URL holding one JSON blob per parent chunk (only
                         set by chunkers that produce parents), e.g.
                         https://<account>.blob.core.windows.net/ingestion-pipeline/parent-chunks/breakdown_policy_booklet/
    chunk_count          number of child chunks written
    parent_count         number of parent chunks written

Child blobs are self-contained: they carry the parent's heading metadata and the
parent's blob URL, so the EmbeddingEncoder can build a search document from a
child alone, without opening any parent blob.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import os

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from storage import (
    BlobLocation,
    delete_blobs,
    download_blob,
    ensure_container,
    list_blobs,
    parse_blob_url,
    upload_json,
)
from ..handler import PipelineContext, PipelineHandler

HEADER_LEVELS = ("H1", "H2", "H3", "H4")


def deterministic_id(*parts: object) -> str:
    """
    Stable 32-char hex id derived from the given parts.

    Deterministic rather than random so that re-ingesting a document overwrites
    its chunks in place — in blob storage and, downstream, in the search index —
    instead of accumulating a second copy alongside the first.
    """
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def breadcrumb_of(metadata: dict) -> str:
    """"Your Cover > Section A. Roadside" from MarkdownHeaderTextSplitter metadata."""
    return " > ".join(v for v in (metadata.get(h, "") for h in HEADER_LEVELS) if v)


@dataclass
class ParentChunk:
    parent_id: str
    content: str
    metadata: dict = field(default_factory=dict)

    def to_payload(self, document_name: str) -> dict:
        # Key names match what rag_retriever.build_context() already expects.
        return {
            "parent_id": self.parent_id,
            "document_name": document_name,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class ChildChunk:
    child_id: str
    content: str
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_payload(self, document_name: str, parent_path: str | None) -> dict:
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id or "",
            "parent_path": parent_path or "",
            "document_name": document_name,
            "content": self.content,
            **{f"meta_{h.lower()}": self.metadata.get(h, "") for h in HEADER_LEVELS},
        }


@dataclass
class ChunkSet:
    """What a chunker produces. `parents` is empty for flat chunking strategies."""

    children: list[ChildChunk] = field(default_factory=list)
    parents: list[ParentChunk] = field(default_factory=list)


class DocChunker(PipelineHandler, ABC):
    """
    Base class for chunkers.

    Reads the Markdown named by `clean_markdown_path`, delegates the split to
    `chunk()`, then writes one JSON blob per chunk under a per-document prefix.
    Child chunks are always written; parent chunks only when the strategy
    produces them. Downstream stages therefore only ever need `chunks_path`,
    which is what makes chunker subclasses interchangeable.
    """

    DEFAULT_CHILD_PREFIX = "chunks"
    DEFAULT_PARENT_PREFIX = "parent-chunks"
    CLEAN_SUFFIX = "_clean"

    def __init__(
        self,
        output_container: str | None = None,
        child_prefix: str | None = None,
        parent_prefix: str | None = None,
        prune: bool = True,
        max_workers: int = 16,
    ) -> None:
        """
        output_container: container for the chunks (default: the source container)
        child_prefix:     top-level folder for child chunks (default: chunks)
        parent_prefix:    top-level folder for parent chunks (default: parent-chunks)
        prune:            delete blobs left under the prefixes by an earlier run
        max_workers:      parallel blob uploads
        """
        super().__init__()
        self.output_container = output_container
        self.child_prefix = child_prefix or os.getenv("CHUNK_PREFIX", self.DEFAULT_CHILD_PREFIX)
        self.parent_prefix = parent_prefix or os.getenv(
            "PARENT_CHUNK_PREFIX", self.DEFAULT_PARENT_PREFIX
        )
        self.prune = prune
        self.max_workers = max_workers

    # -- PipelineHandler ----------------------------------------------------

    def process(self, context: PipelineContext) -> PipelineContext:
        source = parse_blob_url(self.require(context, "clean_markdown_path"))
        document_name = context.get("document_name") or self.document_name(source)

        self.log(f"Chunking {source.url}")
        markdown = download_blob(source).decode("utf-8")
        chunk_set = self.chunk(markdown, document_name)

        child_dir = self.chunk_directory(source, self.child_prefix, document_name)
        parent_dir = self.chunk_directory(source, self.parent_prefix, document_name)
        ensure_container(child_dir)
        ensure_container(parent_dir)

        parent_paths: dict[str, str] = {}
        if chunk_set.parents:
            parent_paths = {
                parent.parent_id: parent_dir.child(f"{parent.parent_id}.json").url
                for parent in chunk_set.parents
            }
            self.log(f"Writing {len(chunk_set.parents)} parent chunks to {parent_dir.directory_url}")
            self._upload_all(
                parent_dir,
                {
                    f"{parent.parent_id}.json": parent.to_payload(document_name)
                    for parent in chunk_set.parents
                },
            )
            context["parent_chunks_path"] = parent_dir.directory_url
            context["parent_count"] = len(chunk_set.parents)

        self.log(f"Writing {len(chunk_set.children)} child chunks to {child_dir.directory_url}")
        self._upload_all(
            child_dir,
            {
                f"{child.child_id}.json": child.to_payload(
                    document_name, parent_paths.get(child.parent_id or "")
                )
                for child in chunk_set.children
            },
        )
        context["chunks_path"] = child_dir.directory_url
        context["chunk_count"] = len(chunk_set.children)
        return context

    # -- subclass contract --------------------------------------------------

    @abstractmethod
    def chunk(self, markdown: str, document_name: str) -> ChunkSet:
        """Split `markdown` into chunks."""

    def chunk_directory(self, source: BlobLocation, prefix: str, document_name: str) -> BlobLocation:
        """Per-document prefix under `prefix`, e.g. chunks/breakdown_policy_booklet."""
        return BlobLocation(
            source.account_url,
            self.output_container or source.container,
            f"{prefix.strip('/')}/{document_name}",
        )

    def document_name(self, source: BlobLocation) -> str:
        """Fallback when the context carries no document_name, e.g. running this stage alone."""
        stem = source.stem
        return stem[: -len(self.CLEAN_SUFFIX)] if stem.endswith(self.CLEAN_SUFFIX) else stem

    # -- persistence --------------------------------------------------------

    def _upload_all(self, directory: BlobLocation, payloads: dict[str, dict]) -> None:
        """Write every payload under `directory`, then remove blobs this run did not write."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            list(
                pool.map(
                    lambda item: upload_json(directory.child(item[0]), item[1]),
                    payloads.items(),
                )
            )

        if not self.prune:
            return
        # Chunk ids are deterministic, so a re-run overwrites its own blobs — but a
        # changed chunk size produces fewer chunks, leaving the tail of the previous
        # run behind. Those would otherwise be embedded as phantom chunks.
        stale = [loc for loc in list_blobs(directory) if loc.filename not in payloads]
        if stale:
            self.log(f"Pruning {len(stale)} stale blobs from {directory.directory_url}")
            delete_blobs(stale)


class ParentChildDocChunker(DocChunker):
    """
    Split on Markdown headings into parents, then split each parent into small
    children. Children are embedded and searched; the matching parent supplies
    the wider context to the LLM at query time.
    """

    def __init__(
        self,
        output_container: str | None = None,
        child_prefix: str | None = None,
        parent_prefix: str | None = None,
        prune: bool = True,
        max_workers: int = 16,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
        min_chunk_chars: int = 50,
    ) -> None:
        super().__init__(output_container, child_prefix, parent_prefix, prune, max_workers)
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    def chunk(self, markdown: str, document_name: str) -> ChunkSet:
        parent_docs = self._split_parents(markdown)
        return self._build_parent_child_pairs(parent_docs, document_name)

    def _split_parents(self, markdown: str) -> list:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
                ("####", "H4"),
            ]
        )
        raw_chunks = splitter.split_text(markdown)
        parents = [c for c in raw_chunks if len(c.page_content.strip()) >= self.min_chunk_chars]
        self.log(
            f"Parents: {len(parents)} (dropped {len(raw_chunks) - len(parents)} "
            f"shorter than {self.min_chunk_chars} chars)"
        )
        return parents

    def _build_parent_child_pairs(self, parent_docs: list, document_name: str) -> ChunkSet:
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
        )
        chunk_set = ChunkSet()

        for parent_index, parent_doc in enumerate(parent_docs):
            metadata = parent_doc.metadata
            breadcrumb = breadcrumb_of(metadata)
            parent_id = deterministic_id(document_name, breadcrumb, parent_index)

            chunk_set.parents.append(
                ParentChunk(
                    parent_id=parent_id,
                    content=parent_doc.page_content,
                    metadata=metadata,
                )
            )

            # Prepend the breadcrumb to each child before embedding so that vector
            # search can match section-specific queries (e.g. "Section D alternative
            # transport") even when the chunk text itself does not name the section.
            # The breadcrumb goes into the embedded child content only — the parent
            # content kept for LLM context stays the original clean text.
            for child_index, text in enumerate(child_splitter.split_text(parent_doc.page_content)):
                chunk_set.children.append(
                    ChildChunk(
                        child_id=deterministic_id(parent_id, child_index),
                        parent_id=parent_id,
                        content=f"{breadcrumb}\n{text}" if breadcrumb else text,
                        metadata=metadata,
                    )
                )

        children = len(chunk_set.children)
        parents = len(chunk_set.parents)
        self.log(f"Children: {children} ({children / max(parents, 1):.1f} per parent)")
        return chunk_set
