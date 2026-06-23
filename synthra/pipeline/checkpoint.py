"""Crash-safe checkpoint storage for generation pipelines.

A checkpoint store durably records which input items have been completed so a
relaunched run can skip them. The default :class:`JsonlCheckpointStore` writes
one record per line and flushes+fsyncs on every write, so the only place
corruption can land is a torn trailing line — which readers discard. That gives
at-least-once execution with exactly-once output (the done-set dedups).

Layout of a checkpoint directory::

    <dir>/
      manifest.json     run metadata, status, config fingerprint, counts
      results.jsonl     completed records: {"id", "ts", "result"}
      errors.jsonl      permanently-failed items: {"id", "ts", "error", "attempts"}
"""

from __future__ import annotations

import abc
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from .blobs import BLOB_KEY, BlobStore

logger = logging.getLogger("synthra.pipeline")


def _collect_blob_hashes(obj: Any, out: set[str]) -> None:
    """Recursively collect sha256 hexes of every blob ref under ``obj``."""
    if BlobStore.is_ref(obj):
        out.add(obj[BLOB_KEY].split(":", 1)[1])
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _collect_blob_hashes(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _collect_blob_hashes(value, out)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Manifest(BaseModel):
    """Run-level metadata persisted alongside the results."""

    name: str
    status: str = "running"  # running | completed | failed
    config_fingerprint: str | None = None
    created_at: str = Field(default_factory=_utcnow)
    updated_at: str = Field(default_factory=_utcnow)
    total: int | None = None
    completed: int = 0
    failed: int = 0
    synthra_version: str | None = None


class CheckpointStore(abc.ABC):
    """Interface for durable, resumable pipeline state."""

    @abc.abstractmethod
    def completed_ids(self) -> set[str]:
        """Return the set of item ids already recorded as done."""

    @abc.abstractmethod
    def append_result(self, item_id: str, result: Any) -> None:
        """Durably append a completed result. Must be safe across threads."""

    @abc.abstractmethod
    def append_error(self, item_id: str, error: str, attempts: int) -> None:
        """Durably append a permanently-failed item."""

    @abc.abstractmethod
    def iter_results(self) -> Iterator[dict]:
        """Yield stored result envelopes ({"id", "ts", "result"})."""

    @abc.abstractmethod
    def load_manifest(self) -> Manifest | None:
        """Load the manifest, or None if this is a fresh run."""

    @abc.abstractmethod
    def save_manifest(self, manifest: Manifest) -> None:
        """Atomically persist the manifest."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Archive any existing state so the run starts fresh."""


class JsonlCheckpointStore(CheckpointStore):
    """Append-only JSONL checkpoint store backed by a local directory."""

    def __init__(self, directory: str | os.PathLike) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        self.results_path = self.dir / "results.jsonl"
        self.errors_path = self.dir / "errors.jsonl"
        self._lock = threading.Lock()
        self._blobs: BlobStore | None = None

    @property
    def blobs(self) -> BlobStore:
        """Content-addressed store for binary payloads, under ``<dir>/blobs``."""
        if self._blobs is None:
            self._blobs = BlobStore(self.dir / "blobs")
        return self._blobs

    # --- reads ------------------------------------------------------------

    def completed_ids(self) -> set[str]:
        ids: set[str] = set()
        for env in self.iter_results():
            item_id = env.get("id")
            if item_id is not None:
                ids.add(item_id)
        return ids

    def results_map(self) -> dict[str, Any]:
        """Return {item_id: result}, deduped by id (keep last)."""
        out: dict[str, Any] = {}
        for env in self.iter_results():
            out[env["id"]] = env.get("result")
        return out

    def referenced_blob_hashes(self) -> set[str]:
        """sha256 hexes of every blob still referenced by a live result."""
        refs: set[str] = set()
        for env in self.iter_results():
            _collect_blob_hashes(env.get("result"), refs)
        return refs

    def gc_blobs(self, *, dry_run: bool = False, min_age_seconds: float = 0.0) -> dict:
        """Delete orphan blobs no live result references. See BlobStore.gc."""
        return self.blobs.gc(
            self.referenced_blob_hashes(),
            dry_run=dry_run,
            min_age_seconds=min_age_seconds,
        )

    def iter_results(self) -> Iterator[dict]:
        if not self.results_path.exists():
            return
        with self.results_path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A torn trailing line after a crash is expected; anything
                    # earlier is suspicious but still skippable.
                    logger.warning(
                        "Skipping unparseable line %d in %s",
                        lineno,
                        self.results_path,
                    )

    # --- writes -----------------------------------------------------------

    def append_result(self, item_id: str, result: Any) -> None:
        self._append(self.results_path, {"id": item_id, "ts": _utcnow(), "result": result})

    def append_error(self, item_id: str, error: str, attempts: int) -> None:
        self._append(
            self.errors_path,
            {"id": item_id, "ts": _utcnow(), "error": error, "attempts": attempts},
        )

    def _append(self, path: Path, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    # --- manifest ---------------------------------------------------------

    def load_manifest(self) -> Manifest | None:
        if not self.manifest_path.exists():
            return None
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return Manifest.model_validate(data)

    def save_manifest(self, manifest: Manifest) -> None:
        manifest.updated_at = _utcnow()
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with self._lock:
            tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp, self.manifest_path)  # atomic on POSIX

    # --- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        existing = [
            p for p in (self.manifest_path, self.results_path, self.errors_path) if p.exists()
        ]
        if not existing:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        archive = self.dir / ".archived" / stamp
        archive.mkdir(parents=True, exist_ok=True)
        for p in existing:
            p.rename(archive / p.name)
        logger.info("Archived previous run state to %s", archive)


def read_results(directory: str | os.PathLike) -> list[Any]:
    """Read the result payloads from a checkpoint dir, deduped by id (keep last)."""
    store = JsonlCheckpointStore(directory)
    by_id: dict[str, Any] = {}
    for env in store.iter_results():
        by_id[env["id"]] = env.get("result")
    return list(by_id.values())


def gc_orphan_blobs(
    directory: str | os.PathLike,
    *,
    dry_run: bool = False,
    min_age_seconds: float = 0.0,
) -> dict:
    """Remove blobs in a checkpoint dir not referenced by any live result.

    Run this against an idle checkpoint (no pipeline writing concurrently), or
    set ``min_age_seconds`` above your longest single-item runtime to avoid
    deleting a blob whose manifest line has not been committed yet. Use
    ``dry_run=True`` to preview.
    """
    return JsonlCheckpointStore(directory).gc_blobs(
        dry_run=dry_run, min_age_seconds=min_age_seconds
    )
