"""Talk to a backend already running under `synthra serve`.

First, in one terminal, start a server and leave it running:

    synthra serve examples/configs/vllm_4gb.yaml
    # or: synthra serve examples/configs/sglang_4gb.yaml

It prints a line like `Ready -> http://127.0.0.1:8000/v1`. Then, in another
terminal, point this client at that URL:

    python examples/client.py
    python examples/client.py --base-url http://127.0.0.1:8001/v1 --model my/model

The server stays up across as many runs as you like — no reload between calls.

Requires the openai client:  uv pip install openai
"""

from __future__ import annotations

import argparse
import time

from openai import OpenAI


def _generate(client: OpenAI, args: argparse.Namespace) -> tuple[str, float, int]:
    """One chat completion; return (text, wall_seconds, completion_tokens)."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0.8,
        max_tokens=args.max_tokens,
    )
    dt = time.perf_counter() - t0
    out = resp.choices[0].message.content.strip()
    completion_tokens = resp.usage.completion_tokens if resp.usage else len(out.split())
    return out, dt, completion_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="The base_url printed by `synthra serve` (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Served model name — the model you launched the server with.",
    )
    parser.add_argument(
        "--api-key",
        default="not-needed",
        help="Only needed if you started the server with --api-key.",
    )
    parser.add_argument(
        "--prompt",
        default="Give me one synthetic customer review for a coffee mug.",
        help="The user message to send.",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Send a warm-up call first (discarded), then measure tokens/s. The "
        "server's own 'Avg throughput' log is averaged over idle time and "
        "understates a single request — this measures the real decode rate.",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.bench:
        # First call pays one-time JIT/CUDA-graph warmup; don't count it.
        print("warming up …")
        _generate(client, args)

    text, dt, n_tokens = _generate(client, args)
    print(text)
    print(
        f"\n[{n_tokens} tokens in {dt:.2f}s = {n_tokens / dt:.1f} tok/s "
        f"(client-measured, end-to-end)]"
    )


if __name__ == "__main__":
    main()
