"""
Stage 2: PreProcessor — clean the extracted Markdown before chunking.

Context in:
    markdown_path        full blob URL of the extracted Markdown, e.g.
                         https://<account>.blob.core.windows.net/ingestion-pipeline/extracted-markdowns/breakdown_policy_booklet.md

Context out:
    clean_markdown_path  full blob URL of the cleaned Markdown, e.g.
                         https://<account>.blob.core.windows.net/ingestion-pipeline/clean-markdowns/breakdown_policy_booklet_clean.md
    clean_markdown       the cleaned Markdown text
"""

from abc import ABC, abstractmethod
import os
import re

from storage import (
    MARKDOWN_CONTENT_TYPE,
    BlobLocation,
    blob_exists,
    download_blob,
    parse_blob_url,
    upload_text,
)
from ..handler import PipelineContext, PipelineHandler


# ---------------------------------------------------------------------------
# Shared cleaning operations
#
# Azure Document Intelligence leaves the same kinds of noise in every booklet —
# page markers, a table of contents, a back cover, logo OCR before the first
# heading — but the specifics differ per document. These are the shared
# operations; each cleaner subclass supplies its own patterns.
# ---------------------------------------------------------------------------

def strip_html_comments(md: str) -> str:
    """Remove page marker comments: <!-- PageBreak -->, <!-- PageNumber=... -->, etc."""
    return re.sub(r'<!--.*?-->\n?', '', md)


def remove_section(md: str, heading: str, stop: str = r'\n# ') -> str:
    """
    Drop a heading and its body, up to the next heading matching `stop`.

    Used for the table of contents, which duplicates every section title and so
    pollutes search results with heading-only matches.
    """
    return re.sub(rf'{heading}\n.*?(?={stop})', '', md, flags=re.DOTALL)


def remove_from(md: str, heading: str) -> str:
    """Drop a heading and everything after it — the back cover and its legal small print."""
    return re.sub(rf'{heading}.*', '', md, flags=re.DOTALL)


def strip_before_first_heading(md: str) -> str:
    """
    Drop logo OCR and cover furniture preceding the first H1.

    Anchored with \\A, not ^: with MULTILINE, ^ matches every line start, which
    would wipe out the content before every H1 in the document.
    """
    return re.sub(r'\A.*?(?=^# )', '', md, flags=re.DOTALL | re.MULTILINE)


def collapse_blank_lines(md: str) -> str:
    return re.sub(r'\n{3,}', '\n\n', md).strip()


def shift_headings(md: str, span: str, frm: str, to: str) -> str:
    """
    Re-level headings inside the region matched by `span`.

    Document Intelligence assigns heading levels per page, so the same kind of
    item — a defined term, say — can come out at different levels in different
    parts of one booklet. Left alone, the chunker treats the odd ones out as
    top-level sections and their breadcrumbs lose the parent heading.
    """
    def relevel(match: re.Match) -> str:
        return match.group(0).replace(f'\n{frm} ', f'\n{to} ')

    return re.sub(span, relevel, md, flags=re.DOTALL)


class PreProcessor(PipelineHandler, ABC):
    """
    Base class for pre-processors.

    A pre-processor reads the Markdown named by `markdown_path`, transforms it,
    writes the result back to blob storage and records its URL in the context
    under `clean_markdown_path`.
    """

    DEFAULT_OUTPUT_PREFIX = "clean-markdowns"
    OUTPUT_SUFFIX = "_clean"

    def __init__(
        self,
        output_container: str | None = None,
        output_prefix: str | None = None,
        skip_if_exists: bool = False,
    ) -> None:
        """
        output_container: container for the cleaned Markdown (default: the source container)
        output_prefix:    virtual folder for the cleaned Markdown (default: clean-markdowns)
        skip_if_exists:   reuse an already-cleaned Markdown instead of re-cleaning
        """
        super().__init__()
        self.output_container = output_container
        self.output_prefix = output_prefix or os.getenv(
            "CLEAN_MARKDOWN_PREFIX", self.DEFAULT_OUTPUT_PREFIX
        )
        self.skip_if_exists = skip_if_exists

    # -- PipelineHandler ----------------------------------------------------

    def process(self, context: PipelineContext) -> PipelineContext:
        source = parse_blob_url(self.require(context, "markdown_path"))
        target = self.clean_markdown_location(source)

        if self.skip_if_exists and blob_exists(target):
            self.log(f"Clean Markdown already exists, reusing {target.url}")
            cleaned = download_blob(target).decode("utf-8")
        else:
            self.log(f"Cleaning {source.url}")
            markdown = download_blob(source).decode("utf-8")
            cleaned = self.clean(markdown)
            removed = len(markdown) - len(cleaned)
            self.log(
                f"{len(markdown):,} -> {len(cleaned):,} chars "
                f"(removed {removed:,}), uploading to {target.url}"
            )
            upload_text(target, cleaned, content_type=MARKDOWN_CONTENT_TYPE)

        context["clean_markdown"] = cleaned
        context["clean_markdown_path"] = target.url
        return context

    # -- subclass contract --------------------------------------------------

    @abstractmethod
    def clean(self, markdown: str) -> str:
        """Return the cleaned version of `markdown`."""

    def clean_markdown_location(self, source: BlobLocation) -> BlobLocation:
        """Where the cleaned Markdown for `source` is written."""
        return source.sibling(
            container=self.output_container or source.container,
            prefix=self.output_prefix,
            filename=f"{source.stem}{self.OUTPUT_SUFFIX}.md",
        )


class MarkdownCleaner(PreProcessor):
    """
    Remove Azure Document Intelligence noise from a policy booklet's Markdown.

    Fix 1: Remove HTML page marker comments (<!-- PageBreak -->, etc.)
    Fix 2: Remove Table of Contents (duplicates all section titles — pollutes search)
    Fix 3a: Remove back cover H1 "# Retirement | Investments ..." and everything after
    Fix 3b: Remove orphaned back cover paragraph via exact str.replace
            (regex with DOTALL would span from cover page to back cover, wiping the document)
    Fix 4: Remove logo OCR text before first heading via \\A anchor
            (^ with MULTILINE would match every line start, wiping content before every H1)
    Fix 5: Normalise #### -> ### inside Definition of words section
            (Azure Doc Intelligence inconsistently assigned heading levels to definition terms)
    """

    # Exact back cover paragraph text — use str.replace, NOT regex, to avoid
    # spanning from the cover page occurrence to the back cover occurrence.
    BACK_COVER_PARA = (
        "If you need breakdown assistance ...\n"
        "call us straight away on 0345 305 7555.\n"
        "For our joint protection calls may be recorded and/or monitored."
    )

    def clean(self, markdown: str) -> str:
        md = strip_html_comments(markdown)                                   # Fix 1
        md = remove_section(md, r'## Contents')                              # Fix 2
        md = remove_from(md, r'\n# Retirement\b')                           # Fix 3a
        md = md.replace(self.BACK_COVER_PARA, '')                            # Fix 3b
        md = strip_before_first_heading(md)                                  # Fix 4
        md = shift_headings(                                                 # Fix 5
            md, r'## Definition of words.*?(?=\n## |\n# )', '####', '###'
        )
        return collapse_blank_lines(md)


class CarInsuranceMarkdownCleaner(PreProcessor):
    """
    Clean the Aviva Zero car insurance policy booklet.

    Same shape of noise as the breakdown booklet, different specifics:

    Fix 1: Remove HTML page markers — this booklet also carries
           <!-- PageHeader="Back to Contents page" --> on nearly every page
    Fix 2: Remove the Table of Contents, which is an H1 here, not an H2
    Fix 3: Remove the back cover "# Insurance Wealth Retirement" and everything
           after it — the FSC logo figure and registration small print
    Fix 4: Remove the <figure> logo OCR block before the first heading
    Fix 5: Demote stray H1 definition terms to H2 inside the Definitions section.
           Most terms come out as H2, but ten of them (Person insured, Schedule,
           Theft, Your car ...) were promoted to H1. Left alone the chunker would
           make each a top-level parent, and "Definitions > Theft" would lose its
           breadcrumb and read as just "Theft".

    Tables are left intact — they carry the cover limits.
    """

    CONTENTS_HEADING = r'# Contents'
    BACK_COVER_HEADING = r'\n# Insurance Wealth Retirement\b'
    # The Definitions section runs until the first numbered policy section.
    DEFINITIONS_SPAN = r'# Definitions\n.*?(?=\n# Section 1\.)'

    def clean(self, markdown: str) -> str:
        md = strip_html_comments(markdown)                                   # Fix 1
        md = remove_section(md, self.CONTENTS_HEADING)                       # Fix 2
        md = remove_from(md, self.BACK_COVER_HEADING)                        # Fix 3
        md = strip_before_first_heading(md)                                  # Fix 4
        md = shift_headings(md, self.DEFINITIONS_SPAN, '#', '##')            # Fix 5
        return collapse_blank_lines(md)
