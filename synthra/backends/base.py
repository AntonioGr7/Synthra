"""Backend server abstraction.

A *backend* is something that serves a model behind an OpenAI-compatible HTTP
API. Synthra launches the server as a subprocess, waits for it to become
healthy, and exposes a ``base_url`` that any OpenAI-compatible client can talk
to. This keeps the rest of the library (generation, heuristics, rules) fully
decoupled from the specific inference engine — today vLLM, tomorrow llama.cpp
or a plain HuggingFace server.
"""

from __future__ import annotations

import abc
import contextlib
import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import IO, Sequence

import httpx
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger("synthra.backends")


def find_free_port(host: str = "127.0.0.1") -> int:
    """Return an OS-assigned free TCP port on ``host``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


class ServerConfig(BaseModel):
    """Configuration shared by every backend server.

    Backend-specific knobs (e.g. ``gpu_memory_utilization`` for vLLM) live on
    the concrete config subclasses; anything not modelled explicitly can be
    passed through ``extra_args``.
    """

    model: str = Field(..., description="Model name or local path to serve.")
    host: str = "127.0.0.1"
    port: int | None = Field(
        default=None,
        description="Port to bind. If None, a free port is chosen at start().",
    )
    api_key: str | None = Field(
        default=None,
        description="API key the server will require (and clients must send).",
    )
    startup_timeout: float = Field(
        default=600.0,
        description="Seconds to wait for the server to become healthy. Model "
        "loading can be slow, so the default is generous.",
    )
    health_poll_interval: float = 1.0
    log_file: Path | None = Field(
        default=None,
        description="If set, server stdout/stderr is written here. Otherwise it "
        "is inherited from the parent process.",
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Raw CLI args appended verbatim to the backend command.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for the server subprocess, e.g. "
        "HF_TOKEN, HF_HOME, CUDA_VISIBLE_DEVICES, or engine tuning vars.",
    )

    model_config = {"extra": "forbid"}

    @classmethod
    def from_dict(cls, data: dict) -> "ServerConfig":
        """Build a config from a mapping, ignoring a top-level ``backend`` key."""
        return cls(**{k: v for k, v in data.items() if k != "backend"})

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "ServerConfig":
        """Load a config from a YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML at {path} must be a mapping, got {type(data)}.")
        return cls.from_dict(data)


class BackendServer(abc.ABC):
    """Manages the lifecycle of an OpenAI-compatible model server subprocess."""

    config: ServerConfig

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._process: subprocess.Popen | None = None
        self._log_handle: IO[bytes] | None = None
        self._resolved_port: int | None = config.port

    # --- subclass contract ------------------------------------------------

    @abc.abstractmethod
    def _build_command(self) -> list[str]:
        """Return the argv used to launch the server.

        Implementations should read ``self.port`` (already resolved) rather than
        ``self.config.port``.
        """

    @property
    def health_path(self) -> str:
        """Relative path of the health endpoint (OpenAI servers use /health)."""
        return "/health"

    # --- derived properties ----------------------------------------------

    @property
    def port(self) -> int:
        if self._resolved_port is None:
            raise RuntimeError("Port not resolved yet; call start() first.")
        return self._resolved_port

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1."""
        return f"http://{self.config.host}:{self.port}/v1"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # --- lifecycle --------------------------------------------------------

    def start(self, wait: bool = True) -> "BackendServer":
        """Launch the server subprocess. If ``wait``, block until healthy."""
        if self.is_running:
            logger.warning("Server already running (pid=%s).", self._process.pid)
            return self

        if self._resolved_port is None:
            self._resolved_port = find_free_port(self.config.host)

        command = self._build_command()
        logger.info("Launching backend: %s", " ".join(command))

        stdout: int | IO[bytes] = subprocess.DEVNULL
        stderr: int | IO[bytes] = subprocess.DEVNULL
        if self.config.log_file is not None:
            self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.config.log_file.open("wb")
            stdout = stderr = self._log_handle
        else:
            # Inherit parent's streams so the user sees model-loading progress.
            stdout = stderr = None  # type: ignore[assignment]

        self._process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            # New session so we can signal the whole process tree (vLLM spawns
            # worker subprocesses) on stop().
            start_new_session=True,
            env={**os.environ, **self._extra_env()},
        )

        if wait:
            self.wait_until_healthy()
        return self

    def _extra_env(self) -> dict[str, str]:
        """Environment overrides for the subprocess.

        Subclasses may extend this; by default it returns the user-supplied
        ``config.env`` (values coerced to strings).
        """
        return {str(k): str(v) for k, v in self.config.env.items()}

    def wait_until_healthy(self) -> None:
        """Block until the server answers its health endpoint or times out."""
        deadline = time.monotonic() + self.config.startup_timeout
        url = f"http://{self.config.host}:{self.port}{self.health_path}"
        headers = self._auth_headers()

        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                code = self._process.returncode
                raise RuntimeError(
                    f"Backend process exited with code {code} before becoming "
                    f"healthy. {self._log_hint()}"
                )
            try:
                resp = httpx.get(url, headers=headers, timeout=5.0)
                if resp.status_code == 200:
                    logger.info("Backend healthy at %s", self.base_url)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(self.config.health_poll_interval)

        self.stop()
        raise TimeoutError(
            f"Backend did not become healthy within "
            f"{self.config.startup_timeout}s. {self._log_hint()}"
        )

    def stop(self, timeout: float = 30.0) -> None:
        """Terminate the server and its process tree."""
        if self._process is None:
            return
        if self._process.poll() is None:
            logger.info("Stopping backend (pid=%s).", self._process.pid)
            self._signal_tree(signal.SIGTERM)
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Backend did not stop gracefully; killing.")
                self._signal_tree(signal.SIGKILL)
                self._process.wait(timeout=timeout)
        self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _signal_tree(self, sig: int) -> None:
        assert self._process is not None
        try:
            os.killpg(os.getpgid(self._process.pid), sig)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                self._process.send_signal(sig)

    # --- helpers ----------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        if self.config.api_key:
            return {"Authorization": f"Bearer {self.config.api_key}"}
        return {}

    def _log_hint(self) -> str:
        if self.config.log_file is not None:
            return f"See logs at {self.config.log_file}."
        return "Server logs were written to this process's stdout/stderr."

    # --- context manager --------------------------------------------------

    def __enter__(self) -> "BackendServer":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
