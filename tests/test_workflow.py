"""Tests for the stage-batched multi-model DAG workflow."""

from __future__ import annotations

from graphlib import CycleError
from types import SimpleNamespace as NS

import pytest

import synthra.workflow.workflow as wfmod
from synthra.workflow import Workflow


# --- DAG: fan-out, fan-in, ordering ---------------------------------------


def test_fanout_converge_correct_judge(tmp_path):
    """The user's shape: image -> {A, B} -> converge -> correct -> judge."""
    order_seen: list[str] = []

    def model_a(item, ctx):
        order_seen.append("a")
        return {"score": item["v"] + 1}

    def model_b(item, ctx):
        order_seen.append("b")
        return {"score": item["v"] * 2}

    def converge(item, ctx):
        # Sees both upstream branch outputs joined by id.
        return {"sum": item["a"]["score"] + item["b"]["score"]}

    def correct(item, ctx):
        return {"sum": item["converge"]["sum"], "fixed": True}

    def judge(item, ctx):
        # Pick whichever branch scored higher.
        winner = "a" if item["a"]["score"] >= item["b"]["score"] else "b"
        return {"winner": winner, "corrected_sum": item["correct"]["sum"]}

    wf = Workflow(str(tmp_path))
    wf.stage("a", model_a)
    wf.stage("b", model_b)
    wf.stage("converge", converge, needs=["a", "b"])
    wf.stage("correct", correct, needs=["converge"])
    wf.stage("judge", judge, needs=["a", "b", "correct"])

    items = [{"id": "x", "v": 3}, {"id": "y", "v": 10}]
    summaries = wf.run(items)

    assert all(s["completed"] == 2 for s in summaries.values())
    # converge ran after both a and b.
    assert set(order_seen) == {"a", "b"}

    res = wf.results("judge")
    # x: a=4, b=6 -> winner b; converge sum=10 -> correct sum=10
    assert res["x"] == {"winner": "b", "corrected_sum": 10}
    # y: a=11, b=20 -> winner b; converge sum=31
    assert res["y"] == {"winner": "b", "corrected_sum": 31}


def test_join_intersects_ids(tmp_path):
    """A converge stage only sees ids present in all upstream branches."""
    def a(item, ctx):
        if item["id"] == "skip":
            raise ValueError("a fails here")
        return {"a": True}

    def b(item, ctx):
        return {"b": True}

    wf = Workflow(str(tmp_path))
    wf.stage("a", a, on_error="skip")
    wf.stage("b", b)
    wf.stage("j", lambda it, ctx: {"joined": True}, needs=["a", "b"])
    summary = wf.run([{"id": "keep"}, {"id": "skip"}])

    assert summary["a"]["completed"] == 1  # "skip" failed
    assert summary["b"]["completed"] == 2
    assert summary["j"]["completed"] == 1  # only "keep" has both branches
    assert set(wf.results("j")) == {"keep"}


def test_cycle_is_detected(tmp_path):
    wf = Workflow(str(tmp_path))
    wf.stage("a", lambda it, ctx: it, needs=["b"])
    wf.stage("b", lambda it, ctx: it, needs=["a"])
    with pytest.raises(CycleError):
        wf.run([{"id": 1}])


def test_unknown_dependency_raises(tmp_path):
    wf = Workflow(str(tmp_path))
    wf.stage("a", lambda it, ctx: it, needs=["nope"])
    with pytest.raises(ValueError, match="unknown stage"):
        wf.run([{"id": 1}])


# --- resume ---------------------------------------------------------------


def test_completed_stages_are_skipped_on_rerun(tmp_path):
    calls = {"a": 0, "b": 0}

    def a(item, ctx):
        calls["a"] += 1
        return {"a": 1}

    def b(item, ctx):
        calls["b"] += 1
        return {"b": item["a"]["a"] + 1}

    def build():
        wf = Workflow(str(tmp_path))
        wf.stage("a", a)
        wf.stage("b", b, needs=["a"])
        return wf

    build().run([{"id": i} for i in range(3)])
    assert calls == {"a": 3, "b": 3}

    # Second run: everything already completed -> no reprocessing.
    summaries = build().run([{"id": i} for i in range(3)])
    assert calls == {"a": 3, "b": 3}
    assert summaries["a"]["skipped_stage"] is True
    assert summaries["b"]["skipped_stage"] is True


# --- shared blobs + GC across stages --------------------------------------


def test_blobs_shared_and_gc_across_stages(tmp_path):
    def produce(item, ctx):
        ref = ctx.blobs.put(bytes([item["id"]]) * 4, media_type="image/png")
        return {"img": ref}

    def consume(item, ctx):
        # Reads the upstream blob through the shared store.
        data = ctx.blobs.get(item["produce"]["img"])
        return {"size": len(data)}

    wf = Workflow(str(tmp_path))
    wf.stage("produce", produce)
    wf.stage("consume", consume, needs=["produce"])
    wf.run([{"id": i} for i in range(3)])

    assert all(r["size"] == 4 for r in wf.results("consume").values())
    # Blobs live in the shared store, referenced by the produce stage.
    gc = wf.gc()
    assert gc["removed"] == 0
    assert gc["orphans"] == 0
    assert gc["referenced"] == gc["total_blobs"] > 0


# --- sequential server lifecycle (fake load_server) -----------------------


class _FakeServer:
    def __init__(self, cfg, events):
        self.cfg = cfg
        self.events = events
        self.config = NS(api_key=None, model=cfg.get("model"))
        self.base_url = f"http://fake/{cfg.get('model')}/v1"

    def start(self):
        self.events.append(("start", self.cfg["model"]))

    def stop(self):
        self.events.append(("stop", self.cfg["model"]))


def test_sequential_server_lifecycle(tmp_path, monkeypatch):
    events: list[tuple] = []
    monkeypatch.setattr(wfmod, "load_server", lambda cfg: _FakeServer(cfg, events))

    A = {"backend": "vllm", "model": "A"}
    B = {"backend": "vllm", "model": "B"}

    wf = Workflow(str(tmp_path))
    wf.stage("s1", lambda it, ctx: {"u": ctx.model}, server=A)
    wf.stage("s2", lambda it, ctx: {"u": ctx.model}, server=A, needs=["s1"])  # reuse A
    wf.stage("s3", lambda it, ctx: {"u": ctx.model}, server=B, needs=["s2"])  # swap to B
    wf.run([{"id": 1}])

    # A starts once (reused for s2), stops when swapping to B; B starts then
    # stops at the end. Exactly one server alive at a time.
    assert events == [
        ("start", "A"),
        ("stop", "A"),
        ("start", "B"),
        ("stop", "B"),
    ]
    # The stage saw the right model via ctx (ids are normalized to strings).
    assert wf.results("s3")["1"]["u"] == "B"


def test_external_base_url_starts_no_server(tmp_path, monkeypatch):
    def boom(cfg):
        raise AssertionError("load_server must not be called for external base_url")

    monkeypatch.setattr(wfmod, "load_server", boom)
    captured = {}

    def stage_fn(item, ctx):
        captured["base_url"] = ctx.base_url
        return {"ok": True}

    wf = Workflow(str(tmp_path))
    wf.stage("ext", stage_fn, server="http://localhost:9999/v1")
    wf.run([{"id": 1}])
    assert captured["base_url"] == "http://localhost:9999/v1"
