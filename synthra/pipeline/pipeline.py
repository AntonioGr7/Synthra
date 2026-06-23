"""Resumable, crash-safe pipeline runner.

A :class:`Pipeline` maps a ``process`` callable over a stream of input items,
checkpointing each completed item so a relaunched run resumes where it stopped.

Contract for resume to work: **the input stream must be deterministic** — the
same items with the same keys across runs. Each item's key (an explicit ``id``
field, a custom ``key`` function, or a content hash) is what identifies it in
the checkpoint, so non-deterministic inputs would skip or duplicate the wrong
records.

Multi-stage pipelines compose by chaining: feed one stage's results
(``read_results(dir)``) as the next stage's input. Each stage checkpoints
independently.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Iterable, Mapping

from .checkpoint import CheckpointStore, JsonlCheckpointStore, Manifest

logger = logging.getLogger("synthra.pipeline")

Item = Any
Record = Any
ProcessFn = Callable[[Item], Record]
KeyFn = Callable[[Item], str]


class PipelineMismatchError(RuntimeError):
    """Raised when resuming a checkpoint whose pipeline fingerprint changed."""


def default_key(item: Item) -> str:
    """Derive a stable key for an item.

    Uses an explicit ``id`` (mapping key or attribute) when present, otherwise a
    content hash of the JSON-serialized item.
    """
    if isinstance(item, Mapping) and "id" in item:
        return str(item["id"])
    if hasattr(item, "id"):
        return str(item.id)
    blob = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class Pipeline:
    """Runs a ``process`` function over items with checkpointing and resume."""

    def __init__(
        self,
        name: str,
        process: ProcessFn,
        checkpoint_dir: str,
        *,
        key: KeyFn | None = None,
        max_workers: int = 8,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        on_error: str = "skip",  # "skip" | "raise"
        fingerprint: str | None = None,
        store: CheckpointStore | None = None,
        log_every: int = 50,
    ) -> None:
        if on_error not in ("skip", "raise"):
            raise ValueError("on_error must be 'skip' or 'raise'.")
        self.name = name
        self.process = process
        self.key_fn = key or default_key
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.on_error = on_error
        self.log_every = log_every
        self.store = store or JsonlCheckpointStore(checkpoint_dir)
        self._fingerprint = fingerprint or self._auto_fingerprint()

    @property
    def blobs(self):
        """Content-addressed blob store for this run (if the store supports it)."""
        blobs = getattr(self.store, "blobs", None)
        if blobs is None:
            raise AttributeError(
                f"{type(self.store).__name__} has no blob store; use a "
                f"JsonlCheckpointStore or pass a BlobStore explicitly."
            )
        return blobs

    def gc(self, *, dry_run: bool = False, min_age_seconds: float = 0.0) -> dict:
        """Delete orphan blobs not referenced by any live result in this run."""
        store_gc = getattr(self.store, "gc_blobs", None)
        if store_gc is None:
            raise AttributeError(
                f"{type(self.store).__name__} does not support blob GC."
            )
        return store_gc(dry_run=dry_run, min_age_seconds=min_age_seconds)

    def _auto_fingerprint(self) -> str:
        """Best-effort fingerprint from the pipeline name and process source."""
        parts = [self.name]
        try:
            parts.append(inspect.getsource(self.process))
        except (OSError, TypeError):
            pass
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]

    # --- run --------------------------------------------------------------

    def run(
        self,
        items: Iterable[Item],
        *,
        resume: bool = True,
        overwrite: bool = False,
        force: bool = False,
        total: int | None = None,
    ) -> dict:
        """Process ``items``, skipping any already checkpointed.

        resume=True (default) skips completed items. overwrite=True archives
        prior state and starts fresh. force=True allows resuming even if the
        pipeline fingerprint changed.
        """
        if overwrite:
            self.store.reset()

        prior = self.store.load_manifest()
        done_ids: set[str] = set()
        if prior is not None and not overwrite:
            if (
                prior.config_fingerprint
                and prior.config_fingerprint != self._fingerprint
                and not force
            ):
                raise PipelineMismatchError(
                    f"Checkpoint at this dir was built by a different pipeline "
                    f"(fingerprint {prior.config_fingerprint} != {self._fingerprint}). "
                    f"Pass force=True to resume anyway, or overwrite=True to restart."
                )
            if resume:
                done_ids = self.store.completed_ids()
            elif prior.completed or prior.failed:
                raise RuntimeError(
                    "Checkpoint already has data but resume=False. Pass "
                    "overwrite=True to discard it or resume=True to continue."
                )

        if total is None and hasattr(items, "__len__"):
            total = len(items)  # type: ignore[arg-type]

        manifest = prior or Manifest(name=self.name)
        manifest.name = self.name
        manifest.status = "running"
        manifest.config_fingerprint = self._fingerprint
        manifest.total = total
        manifest.synthra_version = _synthra_version()
        manifest.completed = len(done_ids)
        self.store.save_manifest(manifest)

        logger.info(
            "Pipeline '%s' starting: %d already done%s.",
            self.name,
            len(done_ids),
            f" of {total}" if total is not None else "",
        )

        skipped = len(done_ids)
        completed = 0
        failed = 0
        seen: set[str] = set(done_ids)

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                in_flight: dict[Future, str] = {}
                cap = self.max_workers * 2
                item_iter = iter(items)
                exhausted = False

                while True:
                    while not exhausted and len(in_flight) < cap:
                        try:
                            item = next(item_iter)
                        except StopIteration:
                            exhausted = True
                            break
                        key = self.key_fn(item)
                        if key in seen:
                            continue  # already done or already queued this run
                        seen.add(key)
                        in_flight[executor.submit(self._run_one, item)] = key

                    if not in_flight:
                        break

                    finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        key = in_flight.pop(fut)
                        try:
                            record = fut.result()
                        except Exception as exc:  # permanent failure
                            failed += 1
                            self.store.append_error(key, str(exc), self.max_retries + 1)
                            logger.warning("Item %s failed permanently: %s", key, exc)
                            if self.on_error == "raise":
                                manifest.completed = len(done_ids) + completed
                                manifest.failed = failed
                                manifest.status = "failed"
                                self.store.save_manifest(manifest)
                                raise
                        else:
                            self.store.append_result(key, record)
                            completed += 1
                            if completed % self.log_every == 0:
                                done_total = len(done_ids) + completed
                                logger.info(
                                    "Pipeline '%s': %d done%s (%d failed).",
                                    self.name,
                                    done_total,
                                    f"/{total}" if total is not None else "",
                                    failed,
                                )
                                manifest.completed = done_total
                                manifest.failed = failed
                                self.store.save_manifest(manifest)

            manifest.completed = len(done_ids) + completed
            manifest.failed = failed
            manifest.status = "completed"
            self.store.save_manifest(manifest)
        except BaseException:
            manifest.completed = len(done_ids) + completed
            manifest.failed = failed
            if manifest.status != "failed":
                manifest.status = "failed"
            self.store.save_manifest(manifest)
            raise

        summary = {
            "name": self.name,
            "completed": len(done_ids) + completed,
            "newly_completed": completed,
            "skipped": skipped,
            "failed": failed,
            "total": total,
        }
        logger.info("Pipeline '%s' finished: %s", self.name, summary)
        return summary

    def _run_one(self, item: Item) -> Record:
        """Run ``process`` with retries and exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.process(item)
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (2**attempt))
        assert last_exc is not None
        raise last_exc


def _synthra_version() -> str | None:
    try:
        from synthra import __version__

        return __version__
    except Exception:
        return None
