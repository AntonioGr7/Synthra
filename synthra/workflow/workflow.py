"""Stage-batched, resumable multi-model DAG workflows.

A :class:`Workflow` is a DAG of stages. Each stage runs a ``process(item, ctx)``
over a stream of items as a resumable :class:`~synthra.pipeline.Pipeline`, so the
machinery we already have (item-level concurrency, checkpointing, retries,
resume, blobs) is reused unchanged. The workflow layer adds three things:

* **Ordering** — stages run in topological order (``graphlib``, stdlib).
* **Joins** — a stage's ``needs`` are joined by item id to form its inputs, so
  fan-out branches (run model A and model B on the same input) reconverge
  cleanly (``{"id", "a": A[id], "b": B[id]}``).
* **Model lifecycle** — *stage-batched, sequential*: bring up a stage's model
  server, sweep the whole dataset through it, then tear it down before the next
  stage needs a different model (reusing it when consecutive stages share a
  config). This amortizes the costly model load across the dataset and fits a
  single GPU — you never need every model resident at once.

Resume falls out for free: a relaunch skips already-completed stages (no server
is even started for them) and resumes partial ones, all keyed by the stable item
id that threads through every stage.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any, Callable, Iterable

from ..backends import load_server
from ..pipeline import BlobStore, Pipeline, default_key
from ..pipeline.checkpoint import JsonlCheckpointStore

logger = logging.getLogger("synthra.workflow")

StageProcess = Callable[[Any, "StageContext"], Any]


@dataclass
class StageContext:
    """Handed to each ``process(item, ctx)`` call."""

    stage: str
    blobs: BlobStore
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    _client: Any = field(default=None, repr=False)

    @property
    def client(self) -> Any:
        """A lazily-built OpenAI client bound to this stage's server (or None)."""
        if self.base_url is None:
            return None
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "openai is required for stages with a server. "
                    "Install with: uv pip install openai"
                ) from exc
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key or "not-needed")
        return self._client


@dataclass
class Stage:
    name: str
    process: StageProcess
    server: Any = None  # yaml path | config dict | "http://..." base_url | None
    needs: tuple[str, ...] = ()
    key: Callable | None = None
    max_workers: int = 8
    max_retries: int = 3
    retry_backoff: float = 1.0
    on_error: str = "skip"


def _is_external(server: Any) -> bool:
    return isinstance(server, str) and server.startswith(("http://", "https://"))


class Workflow:
    """A DAG of model stages over a shared item stream."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # One blob store shared by all stages, so an image stored upstream is
        # referenced (not copied) downstream.
        self.blobs = BlobStore(self.root / "blobs")
        self.stages: dict[str, Stage] = {}

    # --- definition -------------------------------------------------------

    def stage(
        self,
        name: str,
        process: StageProcess,
        *,
        server: Any = None,
        needs: Iterable[str] = (),
        key: Callable | None = None,
        max_workers: int = 8,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        on_error: str = "skip",
    ) -> "Workflow":
        if name in self.stages:
            raise ValueError(f"Duplicate stage name: {name!r}")
        self.stages[name] = Stage(
            name, process, server, tuple(needs), key,
            max_workers, max_retries, retry_backoff, on_error,
        )
        return self

    def order(self) -> list[str]:
        """Topological order of stages (raises graphlib.CycleError on a cycle)."""
        graph = {n: set(s.needs) for n, s in self.stages.items()}
        for name, stage in self.stages.items():
            for dep in stage.needs:
                if dep not in self.stages:
                    raise ValueError(f"Stage {name!r} needs unknown stage {dep!r}.")
        return list(TopologicalSorter(graph).static_order())

    # --- paths ------------------------------------------------------------

    def stage_dir(self, name: str) -> Path:
        return self.root / "stages" / name

    def stage_store(self, name: str) -> JsonlCheckpointStore:
        return JsonlCheckpointStore(self.stage_dir(name))

    def results(self, stage: str) -> dict[str, Any]:
        """Return {id: result} for a finished stage."""
        return self.stage_store(stage).results_map()

    # --- run --------------------------------------------------------------

    def run(self, items: Iterable[Any], *, resume: bool = True, overwrite: bool = False) -> dict:
        order = self.order()
        self._root_items = list(items)
        summaries: dict[str, Any] = {}

        managed_server = None
        current_sig: str | None = None

        try:
            for name in order:
                stage = self.stages[name]
                store = self.stage_store(name)

                manifest = store.load_manifest()
                if manifest is not None and manifest.status == "completed" and not overwrite:
                    logger.info("Stage '%s' already completed; skipping.", name)
                    summaries[name] = {"skipped_stage": True, "completed": manifest.completed}
                    continue

                stage_items = self._build_inputs(stage)
                base_url = api_key = model = None

                if stage.server is not None:
                    if _is_external(stage.server):
                        base_url = stage.server
                    else:
                        sig = self._server_sig(stage.server)
                        if sig != current_sig:
                            if managed_server is not None:
                                managed_server.stop()
                            logger.info("Stage '%s': starting model server.", name)
                            managed_server = load_server(stage.server)
                            managed_server.start()
                            current_sig = sig
                        else:
                            logger.info("Stage '%s': reusing running server.", name)
                        base_url = managed_server.base_url
                        api_key = managed_server.config.api_key
                        model = managed_server.config.model

                ctx = StageContext(
                    stage=name, blobs=self.blobs,
                    base_url=base_url, api_key=api_key, model=model,
                )
                pipe = Pipeline(
                    name,
                    self._wrap(stage, ctx),
                    str(self.stage_dir(name)),
                    key=stage.key or default_key,
                    max_workers=stage.max_workers,
                    max_retries=stage.max_retries,
                    retry_backoff=stage.retry_backoff,
                    on_error=stage.on_error,
                    fingerprint=self._stage_fingerprint(stage),
                    store=store,
                )
                logger.info("Stage '%s': running over %d item(s).", name, len(stage_items))
                summaries[name] = pipe.run(stage_items, resume=resume, overwrite=overwrite)
        finally:
            if managed_server is not None:
                managed_server.stop()

        return summaries

    # --- helpers ----------------------------------------------------------

    def _build_inputs(self, stage: Stage) -> list[Any]:
        if not stage.needs:
            return self._root_items
        maps = {dep: self.stage_store(dep).results_map() for dep in stage.needs}
        # Join on ids present in *every* upstream stage; sorted for determinism
        # (resume requires a stable input order).
        common = set.intersection(*(set(m) for m in maps.values())) if maps else set()
        joined = []
        for item_id in sorted(common):
            item: dict[str, Any] = {"id": item_id}
            for dep in stage.needs:
                item[dep] = maps[dep][item_id]
            joined.append(item)
        return joined

    @staticmethod
    def _wrap(stage: Stage, ctx: StageContext) -> Callable[[Any], Any]:
        process = stage.process

        def run_one(item: Any) -> Any:
            return process(item, ctx)

        return run_one

    @staticmethod
    def _server_sig(server: Any) -> str:
        if isinstance(server, dict):
            blob = json.dumps(server, sort_keys=True, default=str)
            return "dict:" + hashlib.sha256(blob.encode()).hexdigest()[:16]
        return "ref:" + str(server)

    def _stage_fingerprint(self, stage: Stage) -> str:
        parts = [stage.name, "needs:" + ",".join(stage.needs)]
        try:
            parts.append(inspect.getsource(stage.process))
        except (OSError, TypeError):
            pass
        parts.append(self._server_sig(stage.server) if stage.server is not None else "noserver")
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]

    # run() stashes the materialized root items here for _build_inputs.
    _root_items: list[Any] = []

    def gc(self, *, dry_run: bool = False, min_age_seconds: float = 0.0) -> dict:
        """GC the shared blob store against refs from every stage's results."""
        referenced: set[str] = set()
        for name in self.stages:
            referenced |= self.stage_store(name).referenced_blob_hashes()
        return self.blobs.gc(referenced, dry_run=dry_run, min_age_seconds=min_age_seconds)
