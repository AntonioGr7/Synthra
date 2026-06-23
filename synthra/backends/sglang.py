"""SGLang backend.

Launches ``python -m sglang.launch_server`` as a subprocess. Like vLLM, SGLang
exposes an OpenAI-compatible API (``/v1/chat/completions``, ``/v1/completions``,
...) and a ``/health`` endpoint, so once it is healthy the rest of Synthra talks
to it through the exact same OpenAI client — a stage can switch from vLLM to
SGLang by changing only the ``backend`` key in its server config.

The config below mirrors :class:`~synthra.backends.vllm.VLLMConfig` wherever the
two engines share a concept (dtype, parallelism, memory fraction, prefix cache,
seed, tool/reasoning parsers), using SGLang's own flag names. Knobs that have no
SGLang equivalent are intentionally absent; anything not modelled can be passed
verbatim via ``extra_args``. Flag names track recent SGLang (the
``sglang.launch_server`` CLI).
"""

from __future__ import annotations

import sys

from pydantic import Field

from .base import BackendServer, ServerConfig


class SGLangConfig(ServerConfig):
    """SGLang-specific server configuration."""

    served_model_name: str | None = Field(
        default=None,
        description="Name clients use to address the model. Defaults to `model`.",
    )

    # --- model / weights --------------------------------------------------
    dtype: str = Field(
        default="auto",
        description="Weight dtype: auto, bfloat16, float16, float32, half.",
    )
    quantization: str | None = Field(
        default=None, description="Quantization method, e.g. awq, gptq, fp8."
    )
    kv_cache_dtype: str = Field(
        default="auto",
        description="KV cache dtype: auto, fp8_e5m2, fp8_e4m3. fp8 roughly "
        "doubles KV-cache capacity (longer context / more concurrency).",
    )
    context_length: int | None = Field(
        default=None,
        description="Max context length; None lets SGLang read it from the model "
        "config. (vLLM calls this max_model_len.)",
    )
    tokenizer_mode: str = Field(
        default="auto", description="auto or slow."
    )
    trust_remote_code: bool = Field(default=False)
    download_dir: str | None = Field(
        default=None, description="Directory to cache downloaded weights."
    )

    # --- parallelism ------------------------------------------------------
    tensor_parallel_size: int = Field(
        default=1, description="GPUs to shard each model layer across (--tp-size)."
    )
    pipeline_parallel_size: int = Field(
        default=1, description="Pipeline stages to split the model across (--pp-size)."
    )
    data_parallel_size: int = Field(
        default=1,
        description="Replicas of the model to run in parallel (--dp-size). "
        "SGLang-specific; raise for throughput on multi-GPU single-node.",
    )

    # --- memory / scheduling (throughput) ---------------------------------
    mem_fraction_static: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of GPU memory for the static (weights + KV) pool, "
        "SGLang's analogue of vLLM's gpu_memory_utilization. None = SGLang's "
        "adaptive default. Lower it if you hit OOM during capture.",
    )
    cpu_offload_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="GiB of model weights to offload to CPU RAM. Lets a model "
        "that doesn't fit in VRAM run (slower). 0 = no offload.",
    )
    chunked_prefill_size: int | None = Field(
        default=None,
        description="Tokens per prefill chunk so long prefills interleave with "
        "decode. -1 disables chunking. None = SGLang default.",
    )
    max_running_requests: int | None = Field(
        default=None,
        description="Max concurrent requests (batch width), SGLang's analogue of "
        "vLLM's max_num_seqs. Raise for throughput; lower to bound memory.",
    )
    max_total_tokens: int | None = Field(
        default=None,
        description="Cap on tokens in the KV pool across all running requests. "
        "Mostly for debugging / pinning memory; None = SGLang infers it.",
    )
    disable_radix_cache: bool = Field(
        default=False,
        description="Turn OFF prefix (radix) caching, which SGLang enables by "
        "default. Leave False to keep the big win when many samples share a "
        "system prompt / few-shot preamble.",
    )

    # --- reproducibility / determinism ------------------------------------
    seed: int | None = Field(
        default=None,
        description="Engine RNG seed (--random-seed). Set for reproducible data.",
    )

    # --- tool / function calling ------------------------------------------
    tool_call_parser: str | None = Field(
        default=None,
        description="Parser matching the model family: qwen25, llama3, mistral, "
        "deepseekv3, ... Enables tool-call parsing on the OpenAI endpoint.",
    )

    # --- reasoning models -------------------------------------------------
    reasoning_parser: str | None = Field(
        default=None,
        description="Extract reasoning into a separate field: deepseek-r1, qwen3, "
        "glm45, ... ",
    )

    # --- structured output ------------------------------------------------
    grammar_backend: str | None = Field(
        default=None,
        description="Constrained-decoding backend: xgrammar, outlines, llguidance. "
        "None = SGLang default.",
    )

    # --- logging ----------------------------------------------------------
    log_requests: bool = Field(
        default=False,
        description="Log every request (verbose). Off by default for clean runs.",
    )


class SGLangServer(BackendServer):
    """Serves a model with SGLang behind an OpenAI-compatible API."""

    config: SGLangConfig

    def __init__(self, config: SGLangConfig | None = None, **kwargs: object) -> None:
        if config is None:
            config = SGLangConfig(**kwargs)  # type: ignore[arg-type]
        super().__init__(config)

    def _build_command(self) -> list[str]:
        c = self.config
        # SGLang ships no `serve`-style console script; the launcher is a module.
        cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", c.model]

        cmd += ["--host", c.host, "--port", str(self.port)]

        # model / weights
        cmd += ["--dtype", c.dtype, "--kv-cache-dtype", c.kv_cache_dtype]
        cmd += ["--tokenizer-mode", c.tokenizer_mode]
        if c.served_model_name:
            cmd += ["--served-model-name", c.served_model_name]
        if c.context_length is not None:
            cmd += ["--context-length", str(c.context_length)]
        if c.quantization:
            cmd += ["--quantization", c.quantization]
        if c.trust_remote_code:
            cmd += ["--trust-remote-code"]
        if c.download_dir:
            cmd += ["--download-dir", c.download_dir]

        # parallelism
        cmd += ["--tp-size", str(c.tensor_parallel_size)]
        cmd += ["--pp-size", str(c.pipeline_parallel_size)]
        cmd += ["--dp-size", str(c.data_parallel_size)]

        # memory / scheduling
        if c.mem_fraction_static is not None:
            cmd += ["--mem-fraction-static", str(c.mem_fraction_static)]
        if c.cpu_offload_gb > 0:
            cmd += ["--cpu-offload-gb", str(c.cpu_offload_gb)]
        if c.chunked_prefill_size is not None:
            cmd += ["--chunked-prefill-size", str(c.chunked_prefill_size)]
        if c.max_running_requests is not None:
            cmd += ["--max-running-requests", str(c.max_running_requests)]
        if c.max_total_tokens is not None:
            cmd += ["--max-total-tokens", str(c.max_total_tokens)]
        if c.disable_radix_cache:
            cmd += ["--disable-radix-cache"]

        # reproducibility
        if c.seed is not None:
            cmd += ["--random-seed", str(c.seed)]

        # tool calling
        if c.tool_call_parser:
            cmd += ["--tool-call-parser", c.tool_call_parser]

        # reasoning
        if c.reasoning_parser:
            cmd += ["--reasoning-parser", c.reasoning_parser]

        # structured output
        if c.grammar_backend:
            cmd += ["--grammar-backend", c.grammar_backend]

        # auth / logging
        if c.api_key:
            cmd += ["--api-key", c.api_key]
        if c.log_requests:
            cmd += ["--log-requests"]

        cmd += list(c.extra_args)
        return cmd
