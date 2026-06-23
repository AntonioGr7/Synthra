"""Enable `python -m synthra ...` as an alias for the `synthra` CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
