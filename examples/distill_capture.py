"""Off-policy distillation capture: generate with a teacher and save its top-k
logprobs per token, checkpointed and resumable.

    uv pip install -e '.[vllm,distill]' openai
    python examples/distill_capture.py

The student trainer later reads each sample's safetensors blob offline (no
teacher running at train time).
"""

import logging

from openai import OpenAI

from synthra import Pipeline, TeacherLogprobs, load_server, read_results
from synthra.distill import load_from_blob

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT_DIR = "runs/distill"
PROMPTS = [
    {"id": f"q{i}", "prompt": p}
    for i, p in enumerate([
        "Explain what a transformer is in one sentence.",
        "Write a haiku about winter.",
        "What is the capital of France?",
        "Give one tip for writing clean code.",
    ])
]

with load_server("examples/vllm_distill.yaml") as server:
    client = OpenAI(base_url=server.base_url, api_key="not-needed")
    teacher = TeacherLogprobs(client, MODEL, top_k=20)

    def capture(item):
        return teacher.generate(
            [{"role": "user", "content": item["prompt"]}],
            store=pipe.blobs,          # save the logprob arrays in the run's blob store
            temperature=0.7,
            max_tokens=128,
        )

    pipe = Pipeline("distill", capture, CHECKPOINT_DIR, max_workers=4, max_retries=3)
    summary = pipe.run(PROMPTS)        # resumable: rerun to continue after a crash

print("\nSummary:", summary)
for rec in read_results(CHECKPOINT_DIR):
    arrays = load_from_blob(pipe.blobs, rec["teacher"])
    print(f"- {rec['num_tokens']} tokens, topk_logprobs shape {arrays['topk_logprobs'].shape}: "
          f"{rec['text'][:60]!r}")
