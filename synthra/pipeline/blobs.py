"""Content-addressed blob storage for binary payloads (images, audio, ...).

Large binary should not be inlined in the JSONL checkpoint (base64 bloat, slow
resume scans, no dedup). Instead, store each blob as its own file named by the
SHA-256 of its content, and reference it from the JSONL record::

    {"caption": "...", "image": {"$blob": "sha256:ab12...", "media_type": "image/png", "bytes": 20481}}

Writes are crash-safe and idempotent: the blob is written to a temp file,
fsynced, then atomically renamed into place. Because the name is the content
hash, identical blobs collapse to one file (free dedup), and a re-run after a
crash simply re-creates an already-present file harmlessly.

Commit ordering in a pipeline: persist the blob first, *then* append the
manifest line that references it. The manifest line is the single commit point —
a crash before it leaves only a harmless orphan blob.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO, Iterator

logger = logging.getLogger("synthra.pipeline")

BLOB_KEY = "$blob"
_TEMP_PREFIX = ".tmp-"


class BlobStore:
    """A content-addressed store of bytes under a directory."""

    def __init__(self, directory: str | os.PathLike) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    # --- writing ----------------------------------------------------------

    def put(self, data: bytes, media_type: str | None = None) -> dict:
        """Store ``data`` and return a reference dict to embed in a record."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"BlobStore.put expects bytes, got {type(data).__name__}.")
        data = bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        target = self._path_for_hex(digest)

        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=_TEMP_PREFIX)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, target)  # atomic; idempotent if a peer won the race
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

        ref: dict[str, Any] = {BLOB_KEY: f"sha256:{digest}", "bytes": len(data)}
        if media_type:
            ref["media_type"] = media_type
        return ref

    def put_file(self, path: str | os.PathLike, media_type: str | None = None) -> dict:
        return self.put(Path(path).read_bytes(), media_type=media_type)

    # --- reading ----------------------------------------------------------

    def get(self, ref: dict) -> bytes:
        return self.path_for(ref).read_bytes()

    def open(self, ref: dict, mode: str = "rb") -> BinaryIO:
        return self.path_for(ref).open(mode)  # type: ignore[return-value]

    def exists(self, ref: dict) -> bool:
        return self.path_for(ref).exists()

    def path_for(self, ref: dict) -> Path:
        return self._path_for_hex(self._hex_of(ref))

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def is_ref(obj: Any) -> bool:
        return (
            isinstance(obj, dict)
            and isinstance(obj.get(BLOB_KEY), str)
            and obj[BLOB_KEY].startswith("sha256:")
        )

    def iter_blobs(self) -> Iterator[tuple[str, Path]]:
        """Yield ``(sha256_hex, path)`` for every stored blob (skips temp files)."""
        for path in self.dir.rglob("*"):
            if path.is_file() and not path.name.startswith(_TEMP_PREFIX):
                yield path.name, path

    def gc(
        self,
        referenced: set[str],
        *,
        dry_run: bool = False,
        min_age_seconds: float = 0.0,
    ) -> dict:
        """Delete blobs whose hash is not in ``referenced`` (orphans).

        ``referenced`` is a set of sha256 hex strings still pointed at by live
        records. To avoid racing a pipeline that has written a blob but not yet
        committed its manifest line, blobs (and leftover temp files) younger than
        ``min_age_seconds`` are left alone. Run GC against an idle checkpoint, or
        pass a grace period larger than your longest single-item runtime.
        """
        now = time.time()
        total = referenced_present = orphans = removed = skipped_recent = 0
        bytes_freed = 0
        removed_hashes: list[str] = []

        for digest, path in self.iter_blobs():
            total += 1
            if digest in referenced:
                referenced_present += 1
                continue
            orphans += 1
            if min_age_seconds > 0 and (now - self._mtime(path)) < min_age_seconds:
                skipped_recent += 1
                continue
            size = path.stat().st_size
            if not dry_run:
                path.unlink()
            removed += 1
            bytes_freed += size
            removed_hashes.append(digest)

        temp_removed = self._gc_temp_files(now, min_age_seconds, dry_run)
        if not dry_run:
            self._prune_empty_dirs()

        summary = {
            "total_blobs": total,
            "referenced": referenced_present,
            "orphans": orphans,
            "removed": removed,
            "skipped_recent": skipped_recent,
            "temp_removed": temp_removed,
            "bytes_freed": bytes_freed,
            "dry_run": dry_run,
            "removed_hashes": removed_hashes,
        }
        logger.info(
            "Blob GC%s: %d orphan(s), removed %d (%d bytes), kept %d referenced.",
            " (dry run)" if dry_run else "",
            orphans,
            removed,
            bytes_freed,
            referenced_present,
        )
        return summary

    @staticmethod
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return time.time()  # treat as brand new -> protected by grace

    def _gc_temp_files(self, now: float, min_age_seconds: float, dry_run: bool) -> int:
        removed = 0
        for path in self.dir.rglob(f"{_TEMP_PREFIX}*"):
            if not path.is_file():
                continue
            if min_age_seconds > 0 and (now - self._mtime(path)) < min_age_seconds:
                continue
            if not dry_run:
                path.unlink(missing_ok=True)
            removed += 1
        return removed

    def _prune_empty_dirs(self) -> None:
        # Remove now-empty shard directories, deepest first; never the root.
        dirs = sorted(
            (p for p in self.dir.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        )
        for d in dirs:
            try:
                next(d.iterdir())
            except StopIteration:
                d.rmdir()

    def _hex_of(self, ref: dict) -> str:
        if not self.is_ref(ref):
            raise ValueError(f"Not a blob reference: {ref!r}")
        return ref[BLOB_KEY].split(":", 1)[1]

    def _path_for_hex(self, digest: str) -> Path:
        # Shard by the first two byte-pairs to keep directories small.
        return self.dir / digest[:2] / digest[2:4] / digest
