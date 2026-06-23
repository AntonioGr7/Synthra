"""Tests for content-addressed blob storage and WebDataset export."""

from __future__ import annotations

import json
import tarfile

import pytest

from synthra.pipeline import (
    BlobStore,
    JsonlCheckpointStore,
    Pipeline,
    gc_orphan_blobs,
    read_results,
    to_webdataset,
)


# --- BlobStore ------------------------------------------------------------


def test_put_get_roundtrip(tmp_path):
    store = BlobStore(tmp_path)
    ref = store.put(b"\x89PNG fake bytes", media_type="image/png")
    assert ref["$blob"].startswith("sha256:")
    assert ref["bytes"] == len(b"\x89PNG fake bytes")
    assert ref["media_type"] == "image/png"
    assert store.get(ref) == b"\x89PNG fake bytes"
    assert store.exists(ref)


def test_content_addressing_dedups(tmp_path):
    store = BlobStore(tmp_path)
    r1 = store.put(b"same")
    r2 = store.put(b"same")
    assert r1["$blob"] == r2["$blob"]
    assert store.path_for(r1) == store.path_for(r2)
    # Exactly one stored file (plus no leftover temp files).
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) == 1
    assert not any(p.name.startswith(".tmp-") for p in files)


def test_different_content_different_ref(tmp_path):
    store = BlobStore(tmp_path)
    assert store.put(b"a")["$blob"] != store.put(b"b")["$blob"]


def test_put_rejects_non_bytes(tmp_path):
    with pytest.raises(TypeError):
        BlobStore(tmp_path).put("a string")  # type: ignore[arg-type]


def test_is_ref():
    assert BlobStore.is_ref({"$blob": "sha256:abc"})
    assert not BlobStore.is_ref({"$blob": "md5:abc"})
    assert not BlobStore.is_ref({"x": 1})
    assert not BlobStore.is_ref("nope")


def test_put_file(tmp_path):
    src = tmp_path / "img.bin"
    src.write_bytes(b"file-bytes")
    store = BlobStore(tmp_path / "store")
    ref = store.put_file(src, media_type="application/octet-stream")
    assert store.get(ref) == b"file-bytes"


# --- integration with the checkpoint store / pipeline ---------------------


def test_store_exposes_blobs(tmp_path):
    cp = JsonlCheckpointStore(tmp_path)
    ref = cp.blobs.put(b"img", media_type="image/png")
    assert (tmp_path / "blobs").exists()
    assert cp.blobs.get(ref) == b"img"


def test_pipeline_with_blob_refs(tmp_path):
    def process(item):
        ref = pipe.blobs.put(item["data"], media_type="image/png")
        return {"label": item["label"], "image": ref}

    items = [{"id": i, "label": f"l{i}", "data": bytes([i]) * 4} for i in range(3)]
    pipe = Pipeline("img", process, str(tmp_path))
    summary = pipe.run(items)
    assert summary["completed"] == 3

    results = read_results(tmp_path)
    assert len(results) == 3
    # The JSONL holds only a small reference, not the bytes.
    raw = (tmp_path / "results.jsonl").read_text()
    assert "$blob" in raw
    # And the bytes are retrievable via the blob store.
    blobs = pipe.blobs
    recovered = sorted(blobs.get(r["image"]) for r in results)
    assert recovered == sorted(it["data"] for it in items)


# --- garbage collection ---------------------------------------------------


def test_gc_removes_orphans_keeps_referenced(tmp_path):
    store = BlobStore(tmp_path)
    keep = store.put(b"keep-me")
    orphan = store.put(b"orphan")
    keep_hex = keep["$blob"].split(":")[1]

    summary = store.gc({keep_hex})
    assert summary["orphans"] == 1
    assert summary["removed"] == 1
    assert summary["referenced"] == 1
    assert summary["bytes_freed"] == len(b"orphan")
    assert store.exists(keep)
    assert not store.exists(orphan)


def test_gc_dry_run_deletes_nothing(tmp_path):
    store = BlobStore(tmp_path)
    orphan = store.put(b"orphan")
    summary = store.gc(set(), dry_run=True)
    assert summary["orphans"] == 1
    assert summary["removed"] == 1  # would-remove count
    assert summary["dry_run"] is True
    assert store.exists(orphan)  # still there


def test_gc_grace_protects_recent_blobs(tmp_path):
    store = BlobStore(tmp_path)
    orphan = store.put(b"just-written")
    # Huge grace window => the freshly written orphan is protected.
    summary = store.gc(set(), min_age_seconds=3600)
    assert summary["orphans"] == 1
    assert summary["skipped_recent"] == 1
    assert summary["removed"] == 0
    assert store.exists(orphan)


def test_gc_cleans_leftover_temp_files(tmp_path):
    store = BlobStore(tmp_path)
    real = store.put(b"real")
    leftover = tmp_path / ".tmp-interrupted"
    leftover.write_bytes(b"partial")
    real_hex = real["$blob"].split(":")[1]

    summary = store.gc({real_hex})
    assert summary["temp_removed"] == 1
    assert not leftover.exists()
    assert store.exists(real)  # the real blob is untouched


def test_gc_orphan_blobs_via_checkpoint(tmp_path):
    """End-to-end: a record's blob is referenced; an extra blob is orphaned."""
    cp = JsonlCheckpointStore(tmp_path)
    ref = cp.blobs.put(b"referenced", media_type="image/png")
    cp.append_result("a", {"image": ref})
    orphan = cp.blobs.put(b"dangling")

    summary = gc_orphan_blobs(tmp_path)
    assert summary["removed"] == 1
    assert cp.blobs.exists(ref)
    assert not cp.blobs.exists(orphan)


def test_pipeline_gc_after_overwrite(tmp_path):
    """Blobs from a discarded run become orphans and are collectable."""
    def process(item):
        return {"image": pipe.blobs.put(item["data"], media_type="image/png")}

    pipe = Pipeline("g", process, str(tmp_path), fingerprint="v1")
    pipe.run([{"id": i, "data": bytes([i]) * 8} for i in range(3)])

    # Restart fresh: results.jsonl is archived, but the old blobs remain on disk.
    pipe2 = Pipeline("g", process, str(tmp_path), fingerprint="v2")
    pipe2.run([{"id": 99, "data": b"new-data"}], overwrite=True)

    before = list(pipe2.blobs.iter_blobs())
    assert len(before) == 4  # 3 stale + 1 live
    summary = pipe2.gc()
    assert summary["removed"] == 3
    assert summary["referenced"] == 1
    assert len(list(pipe2.blobs.iter_blobs())) == 1


# --- WebDataset export ----------------------------------------------------


def test_to_webdataset_packs_records_and_blobs(tmp_path):
    cp_dir = tmp_path / "cp"
    out_dir = tmp_path / "wds"

    def process(item):
        ref = pipe.blobs.put(item["data"], media_type="image/png")
        return {"caption": item["caption"], "image": ref}

    pipe = Pipeline(
        "wds",
        process,
        str(cp_dir),
    )
    items = [
        {"id": "sample-a", "caption": "a cat", "data": b"AAAA"},
        {"id": "sample-b", "caption": "a dog", "data": b"BBBB"},
    ]
    pipe.run(items)

    summary = to_webdataset(str(cp_dir), str(out_dir))
    assert summary["samples"] == 2
    assert summary["shards"] == 1

    shard = next(out_dir.glob("*.tar"))
    with tarfile.open(shard) as tar:
        names = tar.getnames()
        # json + one image per sample, key sanitized (dots -> underscores).
        assert "sample-a.json" in names
        assert "sample-a.0.png" in names
        assert "sample-b.json" in names
        assert "sample-b.0.png" in names

        meta = json.loads(tar.extractfile("sample-a.json").read())
        # Blob ref was rewritten to point at the in-tar filename.
        assert meta["image"]["file"] == "sample-a.0.png"
        assert tar.extractfile("sample-a.0.png").read() == b"AAAA"


def test_to_webdataset_sharding(tmp_path):
    cp_dir = tmp_path / "cp"
    pipe = Pipeline("s", lambda it: {"v": it["id"]}, str(cp_dir))
    pipe.run([{"id": i} for i in range(5)])
    summary = to_webdataset(str(cp_dir), str(tmp_path / "out"), max_per_shard=2)
    assert summary["samples"] == 5
    assert summary["shards"] == 3  # 2 + 2 + 1
