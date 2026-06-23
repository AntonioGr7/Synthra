"""Backend registry and YAML-driven factory.

A single YAML file fully describes a server. Its ``backend`` key selects the
engine; the remaining keys configure it::

    backend: vllm
    model: Qwen/Qwen2.5-0.5B-Instruct
    port: 8000
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    extra_args: ["--seed", "0"]
"""

from __future__ import annotations

import os

import yaml

from .base import BackendServer, ServerConfig
from .vllm import VLLMConfig, VLLMServer

# name -> (config class, server class)
BACKEND_REGISTRY: dict[str, tuple[type[ServerConfig], type[BackendServer]]] = {
    "vllm": (VLLMConfig, VLLMServer),
}


def register_backend(
    name: str, config_cls: type[ServerConfig], server_cls: type[BackendServer]
) -> None:
    """Register a backend so it can be selected by name in YAML/dicts."""
    BACKEND_REGISTRY[name] = (config_cls, server_cls)


def _resolve(name: str) -> tuple[type[ServerConfig], type[BackendServer]]:
    try:
        return BACKEND_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(BACKEND_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown backend {name!r}. Registered backends: {known}."
        ) from None


def load_server(config: str | os.PathLike | dict) -> BackendServer:
    """Build a backend server from a YAML path or a config mapping.

    The mapping must contain a ``backend`` key naming a registered backend.
    """
    if isinstance(config, dict):
        data = config
    else:
        with open(config, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML at {config} must be a mapping, got {type(data)}.")

    backend = data.get("backend")
    if not backend:
        raise ValueError(
            "Config is missing the 'backend' key (e.g. backend: vllm)."
        )

    config_cls, server_cls = _resolve(backend)
    cfg = config_cls.from_dict(data)
    return server_cls(cfg)
