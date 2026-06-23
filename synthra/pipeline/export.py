"""Export a finished checkpoint to training-friendly formats.

The live checkpoint (JSONL manifest + content-addressed blobs) is optimized for
crash-safe, resumable *writing*. For *training* you usually want sequential
shards. :func:`to_webdataset` packs each record and its blobs into tar shards
following the WebDataset convention: all files for one sample share a key (the
text before the first ``.``), and each component's extension drives decoding.
"""

from __future__ import annotations

import io
import json
import logging
import re
import tarfile
from pathlib import Path
from typing import Any

from .blobs import BlobStore
from .checkpoint import JsonlCheckpointStore

logger = logging.getLogger("synthra.pipeline")

_EXT_BY_MEDIA = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/flac": "flac",
    "video/mp4": "mp4",
    "text/plain": "txt",
    "application/json": "json",
    "application/pdf": "pdf",
}


def _ext_for(media_type: str | None) -> str:
    return _EXT_BY_MEDIA.get((media_type or "").lower(), "bin")


def _sanitize_key(value: Any) -> str:
    # WebDataset splits the sample key on the first '.', so the key must not
    # contain one; keep it filesystem- and tar-safe.
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def to_webdataset(
    checkpoint_dir: str,
    output_dir: str,
    *,
    shard_pattern: str = "shard-{index:06d}.tar",
    max_per_shard: int = 10000,
) -> dict:
    """Pack a checkpoint's results into WebDataset tar shards.

    Each sample becomes ``<key>.json`` (the record, with blob refs rewritten to
    carry the in-tar filename) plus one member per blob (``<key>.<i>.<ext>``).
    Returns a summary dict.
    """
    store = JsonlCheckpointStore(checkpoint_dir)
    blobs = store.blobs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    count_in_shard = 0
    total = 0
    tar: tarfile.TarFile | None = None

    def _add_bytes(t: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))

    try:
        for env in store.iter_results():
            if tar is None or count_in_shard >= max_per_shard:
                if tar is not None:
                    tar.close()
                shard_path = out / shard_pattern.format(index=shard_index)
                tar = tarfile.open(shard_path, "w")
                shard_index += 1
                count_in_shard = 0

            key = _sanitize_key(env["id"])
            blob_members: list[tuple[str, bytes]] = []

            def rewrite(obj: Any) -> Any:
                if BlobStore.is_ref(obj):
                    ext = _ext_for(obj.get("media_type"))
                    member = f"{key}.{len(blob_members)}.{ext}"
                    blob_members.append((member, blobs.get(obj)))
                    enriched = dict(obj)
                    enriched["file"] = member
                    return enriched
                if isinstance(obj, dict):
                    return {k: rewrite(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [rewrite(v) for v in obj]
                return obj

            rewritten = rewrite(env.get("result"))
            payload = json.dumps(rewritten, ensure_ascii=False, default=str).encode("utf-8")
            # json first, then blobs — all members of a key stay contiguous.
            _add_bytes(tar, f"{key}.json", payload)
            for member, data in blob_members:
                _add_bytes(tar, member, data)

            count_in_shard += 1
            total += 1
    finally:
        if tar is not None:
            tar.close()

    summary = {"shards": shard_index, "samples": total, "output_dir": str(out)}
    logger.info("Exported %d samples to %d shard(s) in %s", total, shard_index, out)
    return summary
