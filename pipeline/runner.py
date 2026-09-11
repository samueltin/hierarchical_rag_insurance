"""
Build and run an ingestion chain from a recipe.

A recipe names one implementation per stage:

    {"doc_cracker": "PDFCracker",
     "preprocessor": "CarInsuranceMarkdownCleaner",
     "doc_chunker": "ParentChildDocChunker",
     "embedding_encoder": "AzureOpenAIEncoder"}

This module knows nothing about the document table or ingestion status — it
takes a recipe and a source URL, runs the chain, and reports what happened. The
admin layer wraps it to persist the outcome.
"""

from dataclasses import dataclass, field

from .handler import PipelineContext, PipelineHandler
from .registry import STAGE_ORDER, resolve, validate_recipe


@dataclass
class RunResult:
    """Outcome of one pipeline run."""

    context: PipelineContext = field(default_factory=dict)
    completed: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return not self.error and len(self.completed) == len(STAGE_ORDER)

    @property
    def stage_reached(self) -> str:
        """The last stage that finished, or "" if the first one failed."""
        return self.completed[-1] if self.completed else ""

    @property
    def failed_stage(self) -> str:
        """The stage that raised, or "" on success."""
        if not self.error or len(self.completed) >= len(STAGE_ORDER):
            return ""
        return STAGE_ORDER[len(self.completed)]


def build_chain(
    recipe: dict[str, str],
    stage_options: dict[str, dict] | None = None,
) -> tuple[PipelineHandler, dict[PipelineHandler, str]]:
    """
    Wire the four stages into a chain.

    Returns the first handler plus a handler->stage-name map, so a caller can
    tell which stage an observer callback came from.
    """
    validate_recipe(recipe)
    options = stage_options or {}

    handlers = []
    stage_of: dict[PipelineHandler, str] = {}
    for stage in STAGE_ORDER:
        handler = resolve(stage, recipe[stage])(**options.get(stage, {}))
        stage_of[handler] = stage
        handlers.append(handler)

    for current, following in zip(handlers, handlers[1:]):
        current.set_next(following)
    return handlers[0], stage_of


def run(
    recipe: dict[str, str],
    source_url: str,
    stage_options: dict[str, dict] | None = None,
    on_stage=None,
) -> RunResult:
    """
    Run the chain over `source_url`.

    `on_stage(stage_name, context)` is called as each stage completes, so a
    caller can report progress while a run that takes minutes is in flight.

    Failures are returned, not raised: the caller needs to record which stage
    died as much as it needs the error itself.
    """
    result = RunResult()
    try:
        first, stage_of = build_chain(recipe, stage_options)
    except ValueError as exc:
        result.error = str(exc)
        return result

    def observe(handler: PipelineHandler, context: PipelineContext) -> None:
        stage = stage_of[handler]
        result.completed.append(stage)
        if on_stage:
            on_stage(stage, context)

    handler = first
    while handler:
        handler.observer = observe
        handler = handler.next_handler

    try:
        result.context = first.handle({"file_path": source_url})
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result
