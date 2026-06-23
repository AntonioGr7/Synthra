"""vLLM backend.

Launches ``vllm serve`` as a subprocess. vLLM exposes an OpenAI-compatible API
(``/v1/chat/completions``, ``/v1/completions``, ...) and a ``/health`` endpoint,
so once it is healthy the rest of Synthra can talk to it like any OpenAI server.

The config below models the flags that matter most for a synthetic-data pipeline:
reproducibility (``seed``), tool/function calling, reasoning extraction, dense
logprobs (the engine cap; the library requests them per call), and the throughput
knobs you tune to keep a generation run fast. Flag names track recent vLLM
(>=0.23 / the v0.11 CLI line). Anything not modelled can be passed verbatim via
``extra_args``.
"""

from __future__ import annotations

import shutil
import sys

from pydantic import Field

from .base import BackendServer, ServerConfig


class VLLMConfig(ServerConfig):
    """vLLM-specific server configuration."""

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
        description="KV cache dtype: auto, fp8, fp8_e4m3, fp8_e5m2. fp8 roughly "
        "doubles KV-cache capacity (longer context / more concurrency).",
    )
    max_model_len: int | None = Field(
        default=None, description="Max context length; None lets vLLM infer it."
    )
    tokenizer_mode: str = Field(
        default="auto", description="auto, slow, mistral, or custom."
    )
    trust_remote_code: bool = Field(default=False)
    download_dir: str | None = Field(
        default=None, description="Directory to cache downloaded weights."
    )

    # --- parallelism ------------------------------------------------------
    tensor_parallel_size: int = Field(
        default=1, description="GPUs to shard each model layer across."
    )
    pipeline_parallel_size: int = Field(
        default=1, description="Stages to split the model across (multi-node)."
    )

    # --- memory / scheduling (throughput) ---------------------------------
    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Fraction of GPU memory vLLM may use. Higher = larger KV "
        "cache = more concurrency, until you OOM.",
    )
    cpu_offload_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="GiB of model weights to offload to CPU RAM. Lets a model "
        "that doesn't fit in VRAM run (slower). 0 = no offload.",
    )
    block_size: int | None = Field(
        default=None, description="KV cache block size: 1, 8, 16, 32, 64, 128."
    )
    max_num_seqs: int | None = Field(
        default=None,
        description="Max sequences processed per iteration (batch width). "
        "Raise for more throughput; lower to bound memory/latency.",
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        description="Max tokens processed per iteration. Larger favours prefill "
        "throughput; should be >= max_model_len. Pairs with chunked prefill.",
    )
    enable_prefix_caching: bool | None = Field(
        default=None,
        description="Cache shared prompt prefixes. Big win when many samples "
        "share a system prompt / few-shot preamble. None = vLLM default.",
    )
    enable_chunked_prefill: bool | None = Field(
        default=None,
        description="Chunk long prefills so they interleave with decode. "
        "Smooths latency under load. None = vLLM default.",
    )
    enforce_eager: bool = Field(
        default=False,
        description="Disable CUDA graphs. Slower; only for debugging or to save "
        "the memory CUDA graph capture would use.",
    )

    # --- reproducibility / determinism ------------------------------------
    seed: int | None = Field(
        default=None,
        description="Engine RNG seed. Set this for reproducible synthetic data.",
    )

    # --- logprobs (matters for the library) -------------------------------
    max_logprobs: int = Field(
        default=20,
        ge=0,
        description="Server-side cap on logprobs per token. The library requests "
        "top_logprobs per call up to this value; raise it to capture a denser "
        "distribution for scoring/filtering/distillation.",
    )
    return_tokens_as_token_ids: bool = Field(
        default=False,
        description="Return tokens as 'token_id:<id>' strings so logprobs carry "
        "exact vocab ids. Required for distillation capture (TeacherLogprobs).",
    )
    logprobs_mode: str | None = Field(
        default=None,
        description="What logprobs/prompt_logprobs contain: raw_logprobs | "
        "processed_logprobs | raw_logits | processed_logits. Use 'raw_logprobs' "
        "for distillation (teacher distribution before sampling processors).",
    )

    # --- tool / function calling ------------------------------------------
    enable_auto_tool_choice: bool = Field(
        default=False,
        description="Let the model emit tool calls. Requires tool_call_parser.",
    )
    tool_call_parser: str | None = Field(
        default=None,
        description="Parser matching the model family: hermes, llama3_json, "
        "mistral, qwen3_coder, pythonic, ... Required when tool calling is on.",
    )

    # --- reasoning models -------------------------------------------------
    reasoning_parser: str | None = Field(
        default=None,
        description="Extract reasoning into a separate field: deepseek_r1, qwen3, "
        "glm45, granite, mistral, ... (modern vLLM: no separate enable flag).",
    )

    # --- logging ----------------------------------------------------------
    enable_log_requests: bool = Field(
        default=False,
        description="Log every request (verbose). Off by default for clean runs.",
    )

    def model_post_init(self, __context: object) -> None:  # noqa: D401
        if self.enable_auto_tool_choice and not self.tool_call_parser:
            raise ValueError(
                "enable_auto_tool_choice requires tool_call_parser to be set "
                "(e.g. 'hermes', 'llama3_json', 'mistral')."
            )


class VLLMServer(BackendServer):
    """Serves a model with vLLM behind an OpenAI-compatible API."""

    config: VLLMConfig

    def __init__(self, config: VLLMConfig | None = None, **kwargs: object) -> None:
        if config is None:
            config = VLLMConfig(**kwargs)  # type: ignore[arg-type]
        super().__init__(config)

    def _build_command(self) -> list[str]:
        c = self.config
        # Prefer the `vllm` console script; fall back to `python -m vllm` so it
        # works even when the script isn't on PATH (e.g. some venv setups).
        if shutil.which("vllm"):
            cmd = ["vllm", "serve", c.model]
        else:
            cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", c.model]

        cmd += ["--host", c.host, "--port", str(self.port)]

        # model / weights
        cmd += ["--dtype", c.dtype, "--kv-cache-dtype", c.kv_cache_dtype]
        cmd += ["--tokenizer-mode", c.tokenizer_mode]
        if c.served_model_name:
            cmd += ["--served-model-name", c.served_model_name]
        if c.max_model_len is not None:
            cmd += ["--max-model-len", str(c.max_model_len)]
        if c.quantization:
            cmd += ["--quantization", c.quantization]
        if c.trust_remote_code:
            cmd += ["--trust-remote-code"]
        if c.download_dir:
            cmd += ["--download-dir", c.download_dir]

        # parallelism
        cmd += ["--tensor-parallel-size", str(c.tensor_parallel_size)]
        cmd += ["--pipeline-parallel-size", str(c.pipeline_parallel_size)]

        # memory / scheduling
        cmd += ["--gpu-memory-utilization", str(c.gpu_memory_utilization)]
        if c.cpu_offload_gb > 0:
            cmd += ["--cpu-offload-gb", str(c.cpu_offload_gb)]
        if c.block_size is not None:
            cmd += ["--block-size", str(c.block_size)]
        if c.max_num_seqs is not None:
            cmd += ["--max-num-seqs", str(c.max_num_seqs)]
        if c.max_num_batched_tokens is not None:
            cmd += ["--max-num-batched-tokens", str(c.max_num_batched_tokens)]
        cmd += _tristate("enable-prefix-caching", c.enable_prefix_caching)
        cmd += _tristate("enable-chunked-prefill", c.enable_chunked_prefill)
        if c.enforce_eager:
            cmd += ["--enforce-eager"]

        # reproducibility
        if c.seed is not None:
            cmd += ["--seed", str(c.seed)]

        # logprobs
        cmd += ["--max-logprobs", str(c.max_logprobs)]
        if c.return_tokens_as_token_ids:
            cmd += ["--return-tokens-as-token-ids"]
        if c.logprobs_mode:
            cmd += ["--logprobs-mode", c.logprobs_mode]

        # tool calling
        if c.enable_auto_tool_choice:
            cmd += ["--enable-auto-tool-choice"]
        if c.tool_call_parser:
            cmd += ["--tool-call-parser", c.tool_call_parser]

        # reasoning
        if c.reasoning_parser:
            cmd += ["--reasoning-parser", c.reasoning_parser]

        # auth / logging
        if c.api_key:
            cmd += ["--api-key", c.api_key]
        if c.enable_log_requests:
            cmd += ["--enable-log-requests"]

        cmd += list(c.extra_args)
        return cmd


def _tristate(flag: str, value: bool | None) -> list[str]:
    """Render a vLLM BooleanOptionalAction flag.

    None -> omit (use vLLM default); True -> --flag; False -> --no-flag.
    """
    if value is None:
        return []
    return [f"--{flag}"] if value else [f"--no-{flag}"]
