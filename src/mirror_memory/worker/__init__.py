"""Worker module -- async evolution and snapshot management."""

from mirror_memory.worker.evolution import enqueue_job, evolution_worker, process_pending
from mirror_memory.worker.snapshot import persist_snapshot, promote_shadow_if_ready

__all__ = [
    "enqueue_job",
    "evolution_worker",
    "persist_snapshot",
    "process_pending",
    "promote_shadow_if_ready",
]
