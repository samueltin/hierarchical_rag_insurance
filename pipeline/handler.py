"""
Core Chain of Responsibility plumbing for the RAG ingestion pipeline.

Every stage of the pipeline is a `PipelineHandler`. Handlers are wired together
with `set_next()` and share a single mutable `context` dict that accumulates the
output of each stage:

    PDFCracker  ->  PreProcessor  ->  DocChunker  ->  EmbeddingEncoder
       |               |                 |                |
    markdown        clean markdown     chunks          vectors

Subclasses implement `process()` (the stage's own work) rather than `handle()`
(the chain plumbing), so a stage can never forget to forward the context to the
next handler.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

# Shared context passed down the chain. Keys are documented per stage.
PipelineContext = dict[str, Any]


class PipelineHandler(ABC):
    """Base class for every pipeline stage."""

    def __init__(self) -> None:
        self._next_handler: "PipelineHandler | None" = None
        # Called with (handler, context) after this stage completes. Lets a runner
        # report progress without breaking the chain into separate calls.
        self.observer: "Callable[[PipelineHandler, PipelineContext], None] | None" = None

    # -- chain wiring -------------------------------------------------------

    def set_next(self, handler: "PipelineHandler") -> "PipelineHandler":
        """Attach the next stage. Returns it so calls can be chained fluently."""
        self._next_handler = handler
        return handler

    @property
    def next_handler(self) -> "PipelineHandler | None":
        return self._next_handler

    # -- execution ----------------------------------------------------------

    def handle(self, context: PipelineContext) -> PipelineContext:
        """Run this stage, then pass the context to the next handler."""
        self.log(f"start ({self.name})")
        context = self.process(context)
        if self.observer:
            self.observer(self, context)
        if self._next_handler:
            return self._next_handler.handle(context)
        return context

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Do this stage's work and return the (mutated) context."""

    # -- helpers ------------------------------------------------------------

    @property
    def name(self) -> str:
        return type(self).__name__

    def log(self, message: str) -> None:
        print(f"--> [{self.name}] {message}")

    def require(self, context: PipelineContext, key: str) -> Any:
        """Fetch a required context key, failing loudly when a stage is missing."""
        if key not in context or context[key] is None:
            raise KeyError(
                f"{self.name} requires '{key}' in the pipeline context. "
                f"Available keys: {sorted(context)}"
            )
        return context[key]
