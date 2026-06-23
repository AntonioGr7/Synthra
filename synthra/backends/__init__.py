"""Inference backends for Synthra.

Each backend manages a model server subprocess that exposes an OpenAI-compatible
HTTP API, decoupling the rest of the library from the specific engine.
"""

from .base import BackendServer, ServerConfig, find_free_port
from .factory import BACKEND_REGISTRY, load_server, register_backend
from .sglang import SGLangConfig, SGLangServer
from .vllm import VLLMConfig, VLLMServer

__all__ = [
    "BackendServer",
    "ServerConfig",
    "find_free_port",
    "VLLMConfig",
    "VLLMServer",
    "SGLangConfig",
    "SGLangServer",
    "load_server",
    "register_backend",
    "BACKEND_REGISTRY",
]
