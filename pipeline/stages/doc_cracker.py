"""
Stage 1: DocCracker — turn a source document into Markdown.

Context in:
    file_path        full blob URL of the source document, e.g.
                     https://<account>.blob.core.windows.net/ingestion-pipeline/source-documents/breakdown_policy_booklet.pdf

Context out:
    markdown_path    full blob URL of the extracted Markdown, e.g.
                     https://<account>.blob.core.windows.net/ingestion-pipeline/extracted-markdowns/breakdown_policy_booklet.md
    markdown         the extracted Markdown text
    document_name    source file stem, e.g. "breakdown_policy_booklet"
"""

from abc import ABC, abstractmethod
import io
import os

import fitz  # PyMuPDF
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.identity import DefaultAzureCredential

from storage import (
    MARKDOWN_CONTENT_TYPE,
    BlobLocation,
    blob_exists,
    download_blob,
    parse_blob_url,
    upload_text,
)
from ..handler import PipelineContext, PipelineHandler


class DocCracker(PipelineHandler, ABC):
    """
    Base class for document crackers.

    A cracker reads the source document named by `file_path`, extracts Markdown
    from it, writes the Markdown back to blob storage and records its URL in the
    context under `markdown_path`.
    """

    DEFAULT_OUTPUT_PREFIX = "extracted-markdowns"

    def __init__(
        self,
        output_container: str | None = None,
        output_prefix: str | None = None,
        skip_if_exists: bool = False,
    ) -> None:
        """
        output_container: container for the Markdown (default: the source container)
        output_prefix:    virtual folder for the Markdown (default: extracted-markdowns)
        skip_if_exists:   reuse an already-extracted Markdown instead of re-cracking
        """
        super().__init__()
        self.output_container = output_container
        self.output_prefix = output_prefix or os.getenv(
            "EXTRACTED_MARKDOWN_PREFIX", self.DEFAULT_OUTPUT_PREFIX
        )
        self.skip_if_exists = skip_if_exists

    # -- PipelineHandler ----------------------------------------------------

    def process(self, context: PipelineContext) -> PipelineContext:
        source = parse_blob_url(self.require(context, "file_path"))
        target = self.markdown_location(source)

        if self.skip_if_exists and blob_exists(target):
            self.log(f"Markdown already exists, reusing {target.url}")
            markdown = download_blob(target).decode("utf-8")
        else:
            self.log(f"Extracting Markdown from {source.url}")
            markdown = self.crack(source)
            self.log(f"Extracted {len(markdown):,} chars, uploading to {target.url}")
            upload_text(target, markdown, content_type=MARKDOWN_CONTENT_TYPE)

        context["document_name"] = source.stem
        context["markdown"] = markdown
        context["markdown_path"] = target.url
        return context

    # -- subclass contract --------------------------------------------------

    @abstractmethod
    def crack(self, source: BlobLocation) -> str:
        """Extract and return the Markdown representation of `source`."""

    def markdown_location(self, source: BlobLocation) -> BlobLocation:
        """Where the extracted Markdown for `source` is written."""
        return source.sibling(
            container=self.output_container or source.container,
            prefix=self.output_prefix,
            filename=f"{source.stem}.md",
        )


class PDFCracker(DocCracker):
    """Crack a PDF into Markdown with Azure Document Intelligence (prebuilt-layout)."""

    def __init__(
        self,
        output_container: str | None = None,
        output_prefix: str | None = None,
        skip_if_exists: bool = False,
        model_id: str = "prebuilt-layout",
        page_batch_size: int | None = None,
    ) -> None:
        super().__init__(output_container, output_prefix, skip_if_exists)
        self.model_id = model_id
        self.page_batch_size = page_batch_size or int(os.getenv("DOC_INTEL_PAGE_BATCH", "2"))
        self._client: DocumentIntelligenceClient | None = None

    @property
    def client(self) -> DocumentIntelligenceClient:
        if self._client is None:
            endpoint = os.getenv("DOC_INTEL_ENDPOINT")
            if not endpoint:
                raise ValueError("Missing DOC_INTEL_ENDPOINT in .env")
            self._client = DocumentIntelligenceClient(
                endpoint=endpoint,
                credential=DefaultAzureCredential(),
            )
        return self._client

    def crack(self, source: BlobLocation) -> str:
        pdf_bytes = download_blob(source)

        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            page_count = pdf.page_count

        result = self._analyze(pdf_bytes)
        self.log(f"Pages in PDF: {page_count}, pages analysed: {len(result.pages)}")

        # A single call can be truncated by the service's per-request page limit;
        # fall back to analysing the document in page batches.
        if page_count > len(result.pages):
            self.log(f"Re-analysing in batches of {self.page_batch_size} pages")
            batches: list[str] = []
            for start in range(1, page_count + 1, self.page_batch_size):
                end = min(start + self.page_batch_size - 1, page_count)
                page_spec = f"{start}-{end}" if start != end else str(start)
                batches.append(self._analyze(pdf_bytes, pages=page_spec).content)
            return "\n\n".join(batches)

        return result.content

    def _analyze(self, pdf_bytes: bytes, pages: str | None = None):
        poller = self.client.begin_analyze_document(
            self.model_id,
            body=io.BytesIO(pdf_bytes),
            output_content_format="markdown",
            pages=pages,
        )
        return poller.result()
