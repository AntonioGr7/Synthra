# Synthra

Create synthetic data leveraging other models, heuristics, and rules — built for
finetuning other models.

## Status

Early core. Today Synthra can stand up an inference **backend**: a model server
subprocess that exposes an OpenAI-compatible HTTP API, which the rest of the
library (generation, heuristics, rules) talks to through one uniform client.

Currently shipped:

- **vLLM backend** (`backend: vllm`) — fully YAML-configurable, including tool/
  function calling, reasoning parsers, dense logprobs (engine cap; requested per
  call), and the throughput knobs (`max_num_seqs`, `max_num_batched_tokens`,
  prefix caching, chunked prefill, `kv_cache_dtype`, …). See `examples/vllm.yaml`.

Planned backends: llama.cpp (`llama-server`), HuggingFace/TGI.

## Install

```bash
uv pip install -e .            # core (lightweight)
uv pip install -e '.[vllm]'    # + vLLM (needs a GPU)
```

## Usage

Every server is fully described by a YAML file. `backend:` selects the engine;
the remaining keys configure it (see `examples/vllm.yaml`):

```yaml
backend: vllm
model: Qwen/Qwen2.5-0.5B-Instruct
port: 8000
gpu_memory_utilization: 0.85
max_model_len: 4096
extra_args: ["--seed", "0"]   # anything not modelled is passed to `vllm serve`
```

```python
from openai import OpenAI
from synthra import load_server

with load_server("examples/vllm.yaml") as server:
    client = OpenAI(base_url=server.base_url, api_key="not-needed")
    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[{"role": "user", "content": "Write a synthetic review."}],
    )
    print(resp.choices[0].message.content)
```

The server launches as a subprocess, `start()` blocks until `/health` is green,
and the context manager tears the whole process tree down on exit. You can also
construct configs in code (`VLLMConfig(...)` / `VLLMConfig.from_yaml(path)`) if
you prefer not to use the factory.

## Architecture

```
synthra/
  backends/
    base.py      BackendServer ABC + ServerConfig (YAML/dict loading,
                 process lifecycle, health polling, teardown)
    vllm.py      VLLMServer / VLLMConfig — builds the `vllm serve` command
    factory.py   backend registry + load_server() YAML factory
```

A backend is anything serving a model behind an OpenAI-compatible API. Adding a
new engine = subclass `BackendServer`, add a `ServerConfig` subclass, and
`register_backend("name", ConfigCls, ServerCls)`.
