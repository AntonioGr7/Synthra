"""Tests for the `synthra` CLI.

The config-building logic (YAML vs inline, overrides, `--` passthrough) is tested
directly. The serve lifecycle is tested against a tiny fake backend so it runs
anywhere — no GPU or model download — with Ctrl-C simulated by a sleep that
raises KeyboardInterrupt once the server is healthy.
"""

from __future__ import annotations

import pytest

from synthra.backends import register_backend
from synthra.backends.base import BackendServer, ServerConfig
from synthra.cli import _build_serve_config, build_parser, cmd_serve


def _parse(argv):
    """Mimic main()'s `--` splitting, then parse."""
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    args = build_parser().parse_args(argv)
    args.extra = extra
    return args


# --- config building ------------------------------------------------------


def test_inline_config_with_overrides_and_passthrough():
    args = _parse(
        ["serve", "--backend", "sglang", "--model", "m",
         "--trust-remote-code", "--port", "8001", "--", "--enable-metrics"]
    )
    cfg = _build_serve_config(args)
    assert cfg == {
        "backend": "sglang",
        "model": "m",
        "port": 8001,
        "extra_args": ["--enable-metrics"],
        "trust_remote_code": True,
    }


def test_yaml_config_overrides_win_and_extra_args_append(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("backend: vllm\nmodel: m\nport: 8000\nextra_args: ['--seed', '0']\n")
    args = _parse(["serve", str(p), "--port", "9000", "--", "--quiet"])
    cfg = _build_serve_config(args)
    assert cfg["port"] == 9000  # CLI wins over the file
    assert cfg["extra_args"] == ["--seed", "0", "--quiet"]  # appended, not replaced


def test_inline_requires_backend_and_model():
    args = _parse(["serve", "--backend", "vllm"])  # no --model
    with pytest.raises(SystemExit):
        _build_serve_config(args)


# --- serve lifecycle against an in-memory fake backend --------------------


class _InMemoryServer(BackendServer):
    """Backend that fakes the lifecycle without spawning a process, so the
    serve loop is exercised deterministically (no startup race)."""

    def _build_command(self):  # pragma: no cover - never launched
        return ["true"]

    def start(self, wait: bool = True) -> "BackendServer":
        self._resolved_port = self.config.port or 8000
        self._alive = True
        return self

    @property
    def is_running(self) -> bool:
        return getattr(self, "_alive", False)

    def stop(self, timeout: float = 30.0) -> None:
        self._alive = False


def test_serve_starts_then_stops_on_keyboard_interrupt(monkeypatch):
    register_backend("fake-cli", ServerConfig, _InMemoryServer)

    captured = {}
    from synthra.cli import load_server as real_load

    def spy_load(config):
        srv = real_load(config)
        captured["server"] = srv
        return srv

    monkeypatch.setattr("synthra.cli.load_server", spy_load)

    def _interrupt(*_):  # Ctrl-C the instant the foreground loop sleeps.
        raise KeyboardInterrupt

    monkeypatch.setattr("synthra.cli.time.sleep", _interrupt)

    args = _parse(["serve", "--backend", "fake-cli", "--model", "x", "--port", "8000"])
    rc = cmd_serve(args)
    assert rc == 0
    assert captured["server"].base_url == "http://127.0.0.1:8000/v1"
    assert not captured["server"].is_running  # torn down in finally
