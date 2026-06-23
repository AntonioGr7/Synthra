"""End-to-end: serve a model with vLLM and generate a dataset with checkpointing.

Run it once; kill it midway with Ctrl-C; run it again — it resumes where it
stopped. Run it a third time to completion and it reports everything skipped.

    uv pip install -e '.[vllm]' openai
    python examples/generate_pipeline.py
"""

import logging

from openai import OpenAI

from synthra import Pipeline, load_server, read_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CHECKPOINT_DIR = "runs/reviews"
TOPICS = [
    {"id": "laptop", "product": "a gaming laptop"},
    {"id": "headphones", "product": "noise-cancelling headphones"},
    {"id": "coffee", "product": "a coffee machine"},
    {"id": "mattress", "product": "a memory-foam mattress"},
    {"id": "backpack", "product": "a travel backpack"},
    {"id": "watch", "product": "a smartwatch"},
]

with load_server("examples/configs/vllm_4gb.yaml") as server:
    client = OpenAI(base_url=server.base_url, api_key="not-needed")

    def generate_review(item):
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            messages=[
                {"role": "user", "content": f"Write a short customer review of {item['product']}."}
            ],
            temperature=0.8,
            max_tokens=120,
        )
        return {"product": item["product"], "review": resp.choices[0].message.content}

    pipe = Pipeline(
        name="reviews",
        process=generate_review,
        checkpoint_dir=CHECKPOINT_DIR,
        max_workers=4,        # vLLM batches these concurrent requests
        max_retries=3,
    )
    summary = pipe.run(TOPICS)  # resume=True by default

print("\nSummary:", summary)
print(f"\n{len(read_results(CHECKPOINT_DIR))} records in {CHECKPOINT_DIR}:")
for rec in read_results(CHECKPOINT_DIR):
    print(f"- {rec['product']}: {rec['review'][:80]}...")
