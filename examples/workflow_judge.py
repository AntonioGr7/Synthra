"""A small multi-stage workflow: draft two answers, then let a judge pick the best.

DAG:  question ──▶ draft_a ─┐
                 └▶ draft_b ─┴▶ judge

All three stages use the SAME model server, so it loads once and is reused
across stages (sequential lifecycle). The run is resumable — kill it and rerun,
and completed stages are skipped.

    uv pip install -e '.[vllm]' openai
    python examples/workflow_judge.py
"""

import logging

from synthra import Workflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

SERVER = "examples/vllm_4gb.yaml"   # one server, reused by every stage

QUESTIONS = [
    {"id": "q1", "question": "How do I keep a Python list sorted as I insert items?"},
    {"id": "q2", "question": "What's the difference between a process and a thread?"},
    {"id": "q3", "question": "When should I use a generator instead of a list?"},
]


def _ask(ctx, prompt, **kw):
    resp = ctx.client.chat.completions.create(
        model=ctx.model,
        messages=[{"role": "user", "content": prompt}],
        **kw,
    )
    return resp.choices[0].message.content.strip()


def draft_a(item, ctx):
    # Concise, low-temperature answer.
    text = _ask(ctx, f"Answer concisely in 1-2 sentences:\n{item['question']}",
                temperature=0.2, max_tokens=120)
    return {"answer": text}


def draft_b(item, ctx):
    # More detailed, higher-temperature answer.
    text = _ask(ctx, f"Answer thoroughly with an example:\n{item['question']}",
                temperature=0.9, max_tokens=200)
    return {"answer": text}


def judge(item, ctx):
    a, b = item["draft_a"]["answer"], item["draft_b"]["answer"]
    verdict = _ask(
        ctx,
        "You are judging two answers. Reply with exactly 'A' or 'B' for the better one.\n\n"
        f"Answer A:\n{a}\n\nAnswer B:\n{b}\n\nBetter answer (A or B):",
        temperature=0.0, max_tokens=2,
    )
    winner = "A" if verdict.upper().startswith("A") else "B"
    return {"winner": winner, "chosen": a if winner == "A" else b}


wf = Workflow("runs/judge")
wf.stage("draft_a", draft_a, server=SERVER)
wf.stage("draft_b", draft_b, server=SERVER)              # reuses the same server
wf.stage("judge", judge, server=SERVER, needs=["draft_a", "draft_b"])

summaries = wf.run(QUESTIONS)

print("\nSummaries:", summaries)
print("\nVerdicts:")
results = wf.results("judge")
for qid, q in [(x["id"], x["question"]) for x in QUESTIONS]:
    r = results.get(qid)
    if r:
        print(f"\nQ ({qid}): {q}")
        print(f"  winner: {r['winner']}")
        print(f"  chosen: {r['chosen'][:120]}...")
