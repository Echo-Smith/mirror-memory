"""Extraction module -- K1 deterministic, K2 semantic, K3 synthesis,
and the verification loop (question -> answer judgment)."""

from mirror_memory.extraction.deterministic import extract_claims
from mirror_memory.extraction.pipeline import ExtractionPipeline
from mirror_memory.extraction.semantic import SemanticExtractor
from mirror_memory.extraction.synthesis import Synthesizer
from mirror_memory.extraction.throttle import compute_extraction_value, should_extract
from mirror_memory.extraction.verification import (
    build_verification_question,
    get_question_candidates,
    parse_verdict,
    record_question_injection,
    run_verification_judgment,
)

__all__ = [
    "ExtractionPipeline",
    "SemanticExtractor",
    "Synthesizer",
    "build_verification_question",
    "compute_extraction_value",
    "extract_claims",
    "get_question_candidates",
    "parse_verdict",
    "record_question_injection",
    "run_verification_judgment",
    "should_extract",
]
