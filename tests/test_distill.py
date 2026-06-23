"""Tests for teacher-logprob capture (generate mode)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from synthra.backends import VLLMServer
from synthra.distill import TeacherLogprobs, load_teacher_logprobs
from synthra.pipeline import BlobStore, Pipeline, gc_orphan_blobs, read_results


def _alt(tid, lp):
    return NS(token=f"token_id:{tid}", logprob=lp)


def _make_response(content_text="hello"):
    # Two decoded tokens, top-3 each; logprobs chosen so exp() sums < 1.
    content = [
        NS(token="token_id:5", logprob=-0.7,
           top_logprobs=[_alt(5, -0.7), _alt(7, -1.6), _alt(9, -2.3)]),
        NS(token="token_id:7", logprob=-0.5,
           top_logprobs=[_alt(7, -0.5), _alt(1, -1.5), _alt(2, -2.5)]),
    ]
    choice = NS(
        message=NS(content=content_text),
        logprobs=NS(content=content),
        finish_reason="stop",
    )
    return NS(choices=[choice])


class FakeClient:
    def __init__(self, response):
        self.calls = []
        create = lambda **kw: (self.calls.append(kw), response)[1]  # noqa: E731
        self.chat = NS(completions=NS(create=create))


# --- capture & round-trip -------------------------------------------------


def test_generate_requests_logprobs_and_returns_ref(tmp_path):
    client = FakeClient(_make_response())
    store = BlobStore(tmp_path)
    teacher = TeacherLogprobs(client, "teacher-model", top_k=3, blob_store=store)

    rec = teacher.generate([{"role": "user", "content": "hi"}], temperature=0.8)

    # The call requested logprobs with the right k and passed sampling kwargs.
    call = client.calls[0]
    assert call["logprobs"] is True
    assert call["top_logprobs"] == 3
    assert call["temperature"] == 0.8

    assert rec["text"] == "hello"
    assert rec["num_tokens"] == 2
    assert rec["top_k"] == 3
    assert rec["finish_reason"] == "stop"
    assert BlobStore.is_ref(rec["teacher"])
    assert store.exists(rec["teacher"])


def test_packed_arrays_roundtrip(tmp_path):
    client = FakeClient(_make_response())
    store = BlobStore(tmp_path)
    teacher = TeacherLogprobs(client, "m", top_k=3, blob_store=store)
    rec = teacher.generate([{"role": "user", "content": "hi"}])

    arrays = load_teacher_logprobs(store.get(rec["teacher"]))
    assert arrays["topk_ids"].shape == (2, 3)
    assert arrays["topk_logprobs"].shape == (2, 3)
    assert arrays["topk_logprobs"].dtype == np.float16

    np.testing.assert_array_equal(arrays["topk_ids"][0], [5, 7, 9])
    np.testing.assert_array_equal(arrays["topk_ids"][1], [7, 1, 2])
    np.testing.assert_array_equal(arrays["chosen_ids"], [5, 7])
    assert abs(float(arrays["topk_logprobs"][0, 0]) - (-0.7)) < 1e-2

    # residual = 1 - sum(exp(top-k logprobs))
    expected0 = 1.0 - sum(np.exp([-0.7, -1.6, -2.3]))
    assert abs(float(arrays["residual"][0]) - expected0) < 1e-3


def test_ragged_position_is_padded(tmp_path):
    resp = _make_response()
    # First token only has a single alternative -> remaining slots padded.
    resp.choices[0].logprobs.content[0].top_logprobs = [_alt(5, -0.2)]
    client = FakeClient(resp)
    store = BlobStore(tmp_path)
    rec = TeacherLogprobs(client, "m", top_k=3, blob_store=store).generate([])

    arrays = load_teacher_logprobs(store.get(rec["teacher"]))
    assert arrays["topk_ids"][0, 0] == 5
    assert arrays["topk_ids"][0, 1] == -1  # padded id
    assert arrays["topk_ids"][0, 2] == -1
    # Padded slots must not leak into the residual mass.
    expected = 1.0 - np.exp(-0.2)
    assert abs(float(arrays["residual"][0]) - expected) < 1e-3


def test_rejects_string_tokens_without_token_ids(tmp_path):
    resp = _make_response()
    resp.choices[0].logprobs.content[0].token = "Hello"  # not token_id:<int>
    teacher = TeacherLogprobs(FakeClient(resp), "m", top_k=3, blob_store=BlobStore(tmp_path))
    with pytest.raises(ValueError, match="token_id:"):
        teacher.generate([])


def test_requires_a_store():
    teacher = TeacherLogprobs(FakeClient(_make_response()), "m", top_k=3)
    with pytest.raises(ValueError, match="No blob store"):
        teacher.generate([])


def test_float32_dtype_option(tmp_path):
    store = BlobStore(tmp_path)
    teacher = TeacherLogprobs(FakeClient(_make_response()), "m", top_k=3,
                              blob_store=store, logprobs_dtype="float32")
    rec = teacher.generate([])
    arrays = load_teacher_logprobs(store.get(rec["teacher"]))
    assert arrays["topk_logprobs"].dtype == np.float32


# --- integration with the pipeline ---------------------------------------


def test_pipeline_distillation_capture_and_gc(tmp_path):
    def process(item):
        client = FakeClient(_make_response(content_text=item["prompt"]))
        teacher = TeacherLogprobs(client, "m", top_k=3, blob_store=pipe.blobs)
        return teacher.generate([{"role": "user", "content": item["prompt"]}])

    pipe = Pipeline("distill", process, str(tmp_path))
    items = [{"id": i, "prompt": f"p{i}"} for i in range(4)]
    summary = pipe.run(items)
    assert summary["completed"] == 4

    results = read_results(tmp_path)
    assert len(results) == 4
    # The JSONL holds refs, not arrays; the arrays live in blobs.
    raw = (tmp_path / "results.jsonl").read_text()
    assert "$blob" in raw and "topk_logprobs" not in raw

    # GC must keep every referenced teacher blob (none are orphans). Note:
    # safetensors headers serialize with non-deterministic key order, so blobs
    # are not reliably content-deduped -- but every one on disk is referenced.
    gc = gc_orphan_blobs(tmp_path)
    assert gc["removed"] == 0
    assert gc["orphans"] == 0
    assert gc["referenced"] == gc["total_blobs"] > 0


# --- server flags ---------------------------------------------------------


def test_vllm_distillation_flags():
    server = VLLMServer(
        model="m", port=1, max_logprobs=64,
        return_tokens_as_token_ids=True, logprobs_mode="raw_logprobs",
    )
    cmd = server._build_command()
    assert cmd[cmd.index("--max-logprobs") + 1] == "64"
    assert "--return-tokens-as-token-ids" in cmd
    assert cmd[cmd.index("--logprobs-mode") + 1] == "raw_logprobs"
