"""
Stage registry — maps the stage names stored on a document row to pipeline classes.

Each document row records which implementation to use for each of the four
stages, so a row is effectively a pipeline recipe:

    doc_cracker       PDFCracker
    preprocessor      CarInsuranceMarkdownCleaner
    doc_chunker       ParentChildDocChunker
    embedding_encoder AzureOpenAIEncoder

Names are resolved against concrete subclasses discovered from each stage's
abstract base class. Resolution is deliberately NOT a dotted-path import: the
table is admin-writable, and importing an arbitrary path out of it would turn a
metadata field into code execution.
"""

import inspect

from .stages import doc_chunker, doc_cracker, embedding_encoder, preprocessor

# Column name on the document row -> (module holding the implementations, ABC)
STAGE_BASES = {
    "doc_cracker": (doc_cracker, doc_cracker.DocCracker),
    "preprocessor": (preprocessor, preprocessor.PreProcessor),
    "doc_chunker": (doc_chunker, doc_chunker.DocChunker),
    "embedding_encoder": (embedding_encoder, embedding_encoder.EmbeddingEncoder),
}

STAGE_ORDER = ("doc_cracker", "preprocessor", "doc_chunker", "embedding_encoder")

DEFAULT_RECIPE = {
    "doc_cracker": "PDFCracker",
    "preprocessor": "MarkdownCleaner",
    "doc_chunker": "ParentChildDocChunker",
    "embedding_encoder": "AzureOpenAIEncoder",
}


def implementations(stage: str) -> dict[str, type]:
    """Concrete subclasses available for one stage, by class name."""
    if stage not in STAGE_BASES:
        raise ValueError(f"Unknown stage {stage!r}. Valid stages: {', '.join(STAGE_ORDER)}")
    module, base = STAGE_BASES[stage]
    return {
        cls.__name__: cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, base) and cls is not base and not inspect.isabstract(cls)
    }


def options() -> dict[str, list[str]]:
    """Every stage's available implementations — drives the console's dropdowns."""
    return {stage: sorted(implementations(stage)) for stage in STAGE_ORDER}


def resolve(stage: str, name: str) -> type:
    """Look up one stage implementation by name."""
    available = implementations(stage)
    if name not in available:
        raise ValueError(
            f"Unknown {stage} {name!r}. Available: {', '.join(sorted(available))}"
        )
    return available[name]


def validate_recipe(recipe: dict[str, str]) -> None:
    """Raise if any stage name in `recipe` is not a known implementation."""
    for stage in STAGE_ORDER:
        name = recipe.get(stage)
        if not name:
            raise ValueError(f"Missing {stage} in recipe")
        resolve(stage, name)
