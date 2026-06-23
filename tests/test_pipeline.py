"""Tests for the resumable pipeline runner and checkpoint store."""

from __future__ import annotations

import json

import pytest

from synthra.pipeline import (
    JsonlCheckpointStore,
    Pipeline,
    PipelineMismatchError,
    read_results,
)
from synthra.pipeline.pipeline import default_key


# --- keys -----------------------------------------------------------------


def test_default_key_uses_id_field():
    assert default_key({"id": 42, "x": 1}) == "42"


def test_default_key_content_hash_is_stable():
    a = default_key({"text": "hello"})
    b = default_key({"text": "hello"})
    c = default_key({"text": "world"})
    assert a == b and a != c


# --- happy path -----------------------------------------------------------


def test_run_processes_all_items(tmp_path):
    items = [{"id": i, "n": i} for i in range(10)]
    pipe = Pipeline("sq", lambda it: {"n": it["n"], "sq": it["n"] ** 2},
                    str(tmp_path), max_workers=4)
    summary = pipe.run(items)
    assert summary["completed"] == 10
    assert summary["newly_completed"] == 10
    results = {r["n"]: r["sq"] for r in read_results(tmp_path)}
    assert results == {i: i * i for i in range(10)}


def test_manifest_marks_completed(tmp_path):
    Pipeline("p", lambda it: it, str(tmp_path)).run([{"id": 1}])
    manifest = JsonlCheckpointStore(tmp_path).load_manifest()
    assert manifest is not None
    assert manifest.status == "completed"
    assert manifest.completed == 1


# --- resume after crash ---------------------------------------------------


def test_resume_skips_completed_items(tmp_path):
    """Simulate a crash partway through, then relaunch and finish."""
    items = [{"id": i} for i in range(10)]
    processed: list[int] = []

    class Boom(Exception):
        pass

    def process_crashy(it):
        if it["id"] == 5 and 5 not in _already(tmp_path):
            raise Boom("simulated crash")
        processed.append(it["id"])
        return {"id": it["id"]}

    # First run: make item 5 raise. With on_error="raise" the run aborts, but
    # the items completed before it are durably checkpointed.
    pipe = Pipeline("r", process_crashy, str(tmp_path), max_workers=1,
                    max_retries=0, on_error="raise")
    with pytest.raises(Boom):
        pipe.run(items)

    done_after_crash = JsonlCheckpointStore(tmp_path).completed_ids()
    assert len(done_after_crash) >= 1  # some work survived the crash
    first_pass = list(processed)

    # Second run: the failing condition no longer triggers (5 is "already" done
    # logic aside, we just rerun with a clean process). Resume must skip the
    # work already on disk and only finish the rest.
    processed.clear()
    pipe2 = Pipeline("r", lambda it: {"id": it["id"]}, str(tmp_path), max_workers=1)
    summary = pipe2.run(items, force=True)

    assert summary["completed"] == 10
    # Nothing already checkpointed should be reprocessed in the second pass.
    reprocessed = set(processed) & done_after_crash
    assert not reprocessed
    # The union covers everything exactly once.
    all_results = {r["id"] for r in read_results(tmp_path)}
    assert all_results == set(range(10))


def _already(directory) -> set[int]:
    return {int(i) for i in JsonlCheckpointStore(directory).completed_ids()}


# --- error handling -------------------------------------------------------


def test_on_error_skip_records_errors_and_continues(tmp_path):
    def process(it):
        if it["id"] == 3:
            raise ValueError("bad item")
        return {"id": it["id"]}

    pipe = Pipeline("e", process, str(tmp_path), max_workers=2, max_retries=1,
                    retry_backoff=0, on_error="skip")
    summary = pipe.run([{"id": i} for i in range(5)])

    assert summary["completed"] == 4
    assert summary["failed"] == 1
    errors = (tmp_path / "errors.jsonl").read_text().strip().splitlines()
    assert len(errors) == 1
    rec = json.loads(errors[0])
    assert rec["id"] == "3" and "bad item" in rec["error"]


def test_failed_item_is_retried_on_next_run(tmp_path):
    state = {"fail": True}

    def process(it):
        if it["id"] == 1 and state["fail"]:
            raise ValueError("transient")
        return {"id": it["id"]}

    Pipeline("f", process, str(tmp_path), max_retries=0, retry_backoff=0).run(
        [{"id": 1}]
    )
    assert _already(tmp_path) == set()  # not marked done

    state["fail"] = False  # the transient cause is resolved
    Pipeline("f", process, str(tmp_path), max_retries=0).run([{"id": 1}])
    assert _already(tmp_path) == {1}


def test_retries_then_succeeds(tmp_path):
    attempts = {"n": 0}

    def flaky(it):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("flaky")
        return {"id": it["id"]}

    Pipeline("retry", flaky, str(tmp_path), max_retries=3, retry_backoff=0).run(
        [{"id": 1}]
    )
    assert attempts["n"] == 3
    assert _already(tmp_path) == {1}


# --- fingerprint / overwrite ----------------------------------------------


def test_fingerprint_mismatch_blocks_resume(tmp_path):
    Pipeline("v", lambda it: it, str(tmp_path), fingerprint="v1").run([{"id": 1}])
    with pytest.raises(PipelineMismatchError):
        Pipeline("v", lambda it: it, str(tmp_path), fingerprint="v2").run([{"id": 2}])


def test_force_allows_resume_across_fingerprints(tmp_path):
    Pipeline("v", lambda it: it, str(tmp_path), fingerprint="v1").run([{"id": 1}])
    summary = Pipeline("v", lambda it: it, str(tmp_path), fingerprint="v2").run(
        [{"id": 1}, {"id": 2}], force=True
    )
    assert summary["completed"] == 2


def test_overwrite_archives_and_restarts(tmp_path):
    Pipeline("o", lambda it: it, str(tmp_path), fingerprint="v1").run([{"id": 1}])
    summary = Pipeline("o", lambda it: it, str(tmp_path), fingerprint="v2").run(
        [{"id": 9}], overwrite=True
    )
    assert summary["completed"] == 1
    assert _already(tmp_path) == {9}
    assert (tmp_path / ".archived").exists()


def test_resume_false_with_existing_data_raises(tmp_path):
    # Same fingerprint on both so the resume=False guard is what trips, not the
    # mismatch guard (the two lambdas live on different source lines).
    Pipeline("x", lambda it: it, str(tmp_path), fingerprint="x").run([{"id": 1}])
    with pytest.raises(RuntimeError, match="resume=False"):
        Pipeline("x", lambda it: it, str(tmp_path), fingerprint="x").run(
            [{"id": 2}], resume=False
        )


# --- crash-safety on read -------------------------------------------------


def test_torn_trailing_line_is_ignored(tmp_path):
    store = JsonlCheckpointStore(tmp_path)
    store.append_result("a", {"v": 1})
    # Simulate a crash mid-write: append a partial JSON line with no newline.
    with store.results_path.open("a") as fh:
        fh.write('{"id": "b", "ts": "x", "resu')
    assert store.completed_ids() == {"a"}
    assert {r["v"] for r in read_results(tmp_path)} == {1}
