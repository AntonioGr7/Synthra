"""Tests for the backend server abstraction.

The vLLM command builder is tested directly. The lifecycle (start, health
polling, stop, context manager) is tested against a tiny fake server so it runs
anywhere — no GPU or model download required.
"""

from __future__ import annotations

import sys
import textwrap

import httpx
import pytest

from synthra.backends import VLLMConfig, VLLMServer, find_free_port, load_server
from synthra.backends.base import BackendServer, ServerConfig


# --- vLLM command builder -------------------------------------------------


def test_vllm_command_contains_core_flags():
    server = VLLMServer(model="Qwen/Qwen2.5-0.5B-Instruct", port=8123)
    cmd = server._build_command()
    assert "serve" in cmd
    assert "Qwen/Qwen2.5-0.5B-Instruct" in cmd
    assert cmd[cmd.index("--port") + 1] == "8123"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.9"


def test_vllm_optional_flags_and_passthrough():
    server = VLLMServer(
        model="m",
        port=8001,
        max_model_len=4096,
        quantization="awq",
        trust_remote_code=True,
        api_key="secret",
        extra_args=["--seed", "0"],
    )
    cmd = server._build_command()
    assert cmd[cmd.index("--max-model-len") + 1] == "4096"
    assert cmd[cmd.index("--quantization") + 1] == "awq"
    assert "--trust-remote-code" in cmd
    assert cmd[cmd.index("--api-key") + 1] == "secret"
    assert cmd[-2:] == ["--seed", "0"]


def test_vllm_throughput_and_logprob_flags():
    server = VLLMServer(
        model="m",
        port=8002,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        kv_cache_dtype="fp8",
        max_logprobs=50,
        seed=0,
    )
    cmd = server._build_command()
    assert cmd[cmd.index("--max-num-seqs") + 1] == "256"
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "8192"
    assert cmd[cmd.index("--kv-cache-dtype") + 1] == "fp8"
    assert cmd[cmd.index("--max-logprobs") + 1] == "50"
    assert cmd[cmd.index("--seed") + 1] == "0"


def test_vllm_tristate_caching_flags():
    on = VLLMServer(model="m", port=1, enable_prefix_caching=True,
                    enable_chunked_prefill=True)._build_command()
    assert "--enable-prefix-caching" in on and "--enable-chunked-prefill" in on

    off = VLLMServer(model="m", port=1, enable_prefix_caching=False,
                     enable_chunked_prefill=False)._build_command()
    assert "--no-enable-prefix-caching" in off and "--no-enable-chunked-prefill" in off

    default = VLLMServer(model="m", port=1)._build_command()
    assert not any("prefix-caching" in a for a in default)
    assert not any("chunked-prefill" in a for a in default)


def test_vllm_tool_calling_flags():
    server = VLLMServer(
        model="m", port=1, enable_auto_tool_choice=True, tool_call_parser="hermes"
    )
    cmd = server._build_command()
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"


def test_vllm_tool_calling_requires_parser():
    with pytest.raises(ValueError, match="tool_call_parser"):
        VLLMConfig(model="m", enable_auto_tool_choice=True)


def test_vllm_reasoning_parser_flag():
    cmd = VLLMServer(model="m", port=1, reasoning_parser="qwen3")._build_command()
    assert cmd[cmd.index("--reasoning-parser") + 1] == "qwen3"


def test_base_url_requires_started_port():
    server = VLLMServer(model="m", port=None)
    with pytest.raises(RuntimeError):
        _ = server.base_url


# --- YAML / factory -------------------------------------------------------


def test_load_server_from_yaml(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "backend: vllm\n"
        "model: my/model\n"
        "port: 9001\n"
        "gpu_memory_utilization: 0.5\n"
        "max_model_len: 2048\n"
        "extra_args: ['--seed', '7']\n"
    )
    server = load_server(yaml_path)
    assert isinstance(server, VLLMServer)
    assert server.config.model == "my/model"
    assert server.config.gpu_memory_utilization == 0.5
    cmd = server._build_command()
    assert cmd[cmd.index("--max-model-len") + 1] == "2048"
    assert cmd[cmd.index("--port") + 1] == "9001"


def test_load_server_from_dict():
    server = load_server({"backend": "vllm", "model": "m", "port": 8080})
    assert isinstance(server, VLLMServer)
    assert server.config.port == 8080


def test_load_server_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        load_server({"backend": "nope", "model": "m"})


def test_load_server_missing_backend():
    with pytest.raises(ValueError, match="missing the 'backend'"):
        load_server({"model": "m"})


def test_config_from_yaml_ignores_backend_key(tmp_path):
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text("backend: vllm\nmodel: m\nport: 7000\n")
    cfg = VLLMConfig.from_yaml(yaml_path)
    assert cfg.model == "m" and cfg.port == 7000


# --- lifecycle against a fake server --------------------------------------

_FAKE_SERVER = textwrap.dedent(
    """
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    port = int(sys.argv[1])

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            code = 200 if self.path == "/health" else 404
            self.send_response(code)
            self.end_headers()
        def log_message(self, *a):
            pass

    HTTPServer(("127.0.0.1", port), H).serve_forever()
    """
)


class FakeServer(BackendServer):
    """Backend that launches the trivial health-serving script above."""

    def _build_command(self):
        return [sys.executable, "-c", _FAKE_SERVER, str(self.port)]


def test_lifecycle_start_health_stop(tmp_path):
    cfg = ServerConfig(
        model="fake",
        port=find_free_port(),
        startup_timeout=15,
        health_poll_interval=0.1,
        log_file=tmp_path / "server.log",
    )
    server = FakeServer(cfg)
    server.start()
    try:
        assert server.is_running
        resp = httpx.get(f"http://127.0.0.1:{server.port}/health", timeout=5)
        assert resp.status_code == 200
    finally:
        server.stop()
    assert not server.is_running


def test_context_manager_picks_free_port():
    cfg = ServerConfig(model="fake", port=None, startup_timeout=15, health_poll_interval=0.1)
    with FakeServer(cfg) as server:
        assert server.is_running
        assert server.port > 0
        assert server.base_url.endswith(f":{server.port}/v1")
    assert not server.is_running


def test_timeout_when_never_healthy(tmp_path):
    # Sleep forever, never serve /health -> should time out and be torn down.
    script = "import time\nwhile True: time.sleep(1)"

    class DeadServer(BackendServer):
        def _build_command(self):
            return [sys.executable, "-c", script]

    cfg = ServerConfig(
        model="fake", port=find_free_port(), startup_timeout=1.5, health_poll_interval=0.2
    )
    server = DeadServer(cfg)
    with pytest.raises(TimeoutError):
        server.start()
    assert not server.is_running
