"""Command-line entry point for Synthra.

The ``serve`` subcommand stands up a single backend from a YAML config (or an
inline ``--backend``/``--model`` pair) and keeps it running until you press
Ctrl-C. This is the quick-iteration path: load a model once, poke it with any
OpenAI client, and tear it down once — instead of letting a workflow load and
unload it on every run.

    synthra serve examples/configs/vllm_4gb.yaml
    synthra serve examples/configs/sglang_4gb.yaml --port 8001
    synthra serve --backend sglang --model baidu/Unlimited-OCR --trust-remote-code

Point a workflow stage at the already-running server with its URL so the
workflow reuses it instead of managing the lifecycle::

    wf.stage("draft", draft, server="http://127.0.0.1:8000/v1")
"""

from __future__ import annotations

import argparse
import sys
import time

from .backends import BACKEND_REGISTRY, load_server


def _build_serve_config(args: argparse.Namespace) -> str | dict:
    """Turn parsed CLI args into a config (YAML path or dict) for load_server."""
    if args.config is not None:
        # A YAML file. Apply the handful of overrides the CLI exposes by reading
        # it into a dict first, so --port/--host/--api-key win over the file.
        import yaml

        with open(args.config, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise SystemExit(f"Config {args.config} must be a YAML mapping.")
        _apply_overrides(data, args)
        return data

    if not args.backend or not args.model:
        raise SystemExit(
            "Provide a config file, or both --backend and --model.\n"
            f"Registered backends: {', '.join(sorted(BACKEND_REGISTRY))}."
        )
    data = {"backend": args.backend, "model": args.model}
    _apply_overrides(data, args)
    if args.trust_remote_code:
        data["trust_remote_code"] = True
    return data


def _apply_overrides(data: dict, args: argparse.Namespace) -> None:
    if args.port is not None:
        data["port"] = args.port
    if args.host is not None:
        data["host"] = args.host
    if args.api_key is not None:
        data["api_key"] = args.api_key
    if args.extra:
        # Append rather than replace any extra_args already in the file.
        data["extra_args"] = list(data.get("extra_args", [])) + args.extra


def cmd_serve(args: argparse.Namespace) -> int:
    config = _build_serve_config(args)
    server = load_server(config)

    print(f"Starting {type(server).__name__} (model={server.config.model}) …",
          file=sys.stderr, flush=True)
    server.start()  # blocks until /health is green or it times out
    served = getattr(server.config, "served_model_name", None) or server.config.model
    print(
        f"\nReady → {server.base_url}\n"
        f"  model name: {served}\n"
        f"  health:     {server.base_url.rsplit('/v1', 1)[0]}{server.health_path}\n"
        "Press Ctrl-C to stop.\n",
        file=sys.stderr,
        flush=True,
    )

    # Stay in the foreground until interrupted or the server dies on its own.
    try:
        while server.is_running:
            time.sleep(0.5)
        print("Backend process exited.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopping backend …", file=sys.stderr, flush=True)
        return 0
    finally:
        server.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthra", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser(
        "serve",
        help="Start one backend and keep it alive until Ctrl-C.",
        description="Start a single backend server from a config or inline flags.",
    )
    serve.add_argument(
        "config",
        nargs="?",
        help="Path to a backend YAML config. Omit to use --backend/--model.",
    )
    serve.add_argument("--backend", help=f"One of: {', '.join(sorted(BACKEND_REGISTRY))}.")
    serve.add_argument("--model", help="Model name or local path to serve.")
    serve.add_argument("--port", type=int, help="Override the bind port.")
    serve.add_argument("--host", help="Override the bind host.")
    serve.add_argument("--api-key", help="API key the server should require.")
    serve.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="(inline mode) Allow loading custom model code, e.g. for OCR/VLMs.",
    )
    serve.set_defaults(func=cmd_serve, extra=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Everything after a literal `--` is passed verbatim to the engine; split it
    # off before argparse so it can't be mistaken for the `config` positional.
    extra: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        argv, extra = argv[:idx], argv[idx + 1 :]

    args = build_parser().parse_args(argv)
    args.extra = extra
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
