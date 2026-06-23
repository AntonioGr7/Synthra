"""Capture a teacher model's top-k logprobs for off-policy distillation.

Saving *full* logits is infeasible (~300 KB per token at a 150k vocab → terabytes
for a real corpus). Instead we keep the **top-k** token ids and logprobs per
position — the softmax is peaky, so the top-k carries almost all the KD signal at
thousands of times less storage. Each sample's arrays are packed into a
``safetensors`` blob and stored via the content-addressed :class:`BlobStore`, so
the JSONL manifest stays tiny and resume/GC/export keep working.

Generate mode: the teacher samples the sequence and we record its top-k
distribution at each decoded position in the same call. For a faithful teacher
distribution, run the server with ``logprobs_mode='raw_logprobs'`` (before
sampling-time processors) and ``return_tokens_as_token_ids=True`` (so logprobs
carry exact vocab ids rather than detokenized strings).

The student trainer can read the blob back offline — no second model running at
train time.
"""

from __future__ import annotations

import importlib
from typing import Any

_TOKEN_ID_PREFIX = "token_id:"
SCHEMA = "synthra.teacher_logprobs.v1"
MEDIA_TYPE = "application/x-safetensors"

_PAD_ID = -1
_PAD_LOGPROB = float("-inf")  # sentinel for ragged positions with fewer than k entries


def _np():
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:  # pragma: no cover - exercised via install state
        raise ImportError(
            "numpy is required for distillation capture. Install with: "
            "uv pip install 'synthra[distill]'"
        ) from exc


def _safetensors():
    try:
        return importlib.import_module("safetensors.numpy")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "safetensors is required for distillation capture. Install with: "
            "uv pip install 'synthra[distill]'"
        ) from exc


def _parse_token_id(token: str) -> int:
    if isinstance(token, str) and token.startswith(_TOKEN_ID_PREFIX):
        return int(token[len(_TOKEN_ID_PREFIX) :])
    raise ValueError(
        f"Expected a 'token_id:<int>' token but got {token!r}. Start the vLLM "
        f"server with return_tokens_as_token_ids=True so logprobs carry vocab ids."
    )


class TeacherLogprobs:
    """Generate with a teacher and capture its top-k logprobs per token."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        top_k: int = 20,
        blob_store: Any | None = None,
        store_chosen: bool = True,
        store_residual: bool = True,
        logprobs_dtype: str = "float16",
    ) -> None:
        if logprobs_dtype not in ("float16", "float32"):
            raise ValueError("logprobs_dtype must be 'float16' or 'float32'.")
        self.client = client
        self.model = model
        self.top_k = top_k
        self.blob_store = blob_store
        self.store_chosen = store_chosen
        self.store_residual = store_residual
        self.logprobs_dtype = logprobs_dtype

    def generate(
        self,
        messages: list[dict],
        *,
        store: Any | None = None,
        **sampling_kwargs: Any,
    ) -> dict:
        """Generate a completion and store its teacher logprobs.

        Returns a record (suitable as a Pipeline result) with the generated text
        and a blob reference under ``"teacher"``. Extra kwargs (temperature,
        max_tokens, ...) pass through to the chat completion.
        """
        store = store or self.blob_store
        if store is None:
            raise ValueError("No blob store provided (set blob_store or pass store=).")

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            top_logprobs=self.top_k,
            **sampling_kwargs,
        )
        choice = resp.choices[0]
        content = choice.logprobs.content if choice.logprobs else []
        blob, seq_len = self._pack(content)
        ref = store.put(blob, media_type=MEDIA_TYPE)

        return {
            "text": choice.message.content,
            "num_tokens": seq_len,
            "top_k": self.top_k,
            "finish_reason": getattr(choice, "finish_reason", None),
            "teacher": ref,
        }

    def _pack(self, content: list) -> tuple[bytes, int]:
        np = _np()
        st = _safetensors()
        seq = len(content)
        k = self.top_k

        ids = np.full((seq, k), _PAD_ID, dtype=np.int32)
        logprobs = np.full((seq, k), _PAD_LOGPROB, dtype=np.float32)
        chosen = np.full(seq, _PAD_ID, dtype=np.int32)

        for i, tok in enumerate(content):
            chosen[i] = _parse_token_id(tok.token)
            for j, alt in enumerate((tok.top_logprobs or [])[:k]):
                ids[i, j] = _parse_token_id(alt.token)
                logprobs[i, j] = alt.logprob

        out_dtype = np.float16 if self.logprobs_dtype == "float16" else np.float32
        tensors = {
            "topk_ids": ids,
            "topk_logprobs": logprobs.astype(out_dtype),
        }
        if self.store_chosen:
            tensors["chosen_ids"] = chosen
        if self.store_residual:
            # Tail mass not covered by the top-k (assumes normalized logprobs).
            probs = np.exp(logprobs.astype(np.float64))
            probs[~np.isfinite(logprobs)] = 0.0  # ignore padded slots
            residual = np.clip(1.0 - probs.sum(axis=1), 0.0, 1.0).astype(np.float32)
            tensors["residual"] = residual

        metadata = {
            "schema": SCHEMA,
            "model": self.model,
            "top_k": str(k),
            "logprobs_dtype": self.logprobs_dtype,
        }
        return st.save(tensors, metadata=metadata), seq


def load_teacher_logprobs(data: bytes) -> dict:
    """Decode a teacher-logprob blob back into numpy arrays."""
    st = _safetensors()
    return st.load(data)


def load_from_blob(store: Any, ref: dict) -> dict:
    """Convenience: read arrays directly from a blob store + reference."""
    return load_teacher_logprobs(store.get(ref))
