"""Extraction module -- K1 deterministic, K2 semantic, K3 synthesis."""

from mirror_memory.extraction.deterministic import extract_claims
from mirror_memory.extraction.pipeline import ExtractionPipeline
from mirror_memory.extraction.semantic import SemanticExtractor
from mirror_memory.extraction.synthesis import Synthesizer
from mirror_memory.extraction.throttle import should_extract

__all__ = [
    "ExtractionPipeline",
    "SemanticExtractor",
    "Synthesizer",
    "extract_claims",
    "should_extract",
]
