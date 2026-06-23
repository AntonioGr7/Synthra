"""Start a vLLM server from a YAML config and generate against it.

Requires the vllm extra and a GPU:  uv pip install -e '.[vllm]' openai
"""

from openai import OpenAI

from synthra import load_server

with load_server("examples/configs/vllm_4gb.yaml") as server:
    # server.base_url -> http://127.0.0.1:<port>/v1
    client = OpenAI(base_url=server.base_url, api_key="not-needed")
    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[{"role": "user", "content": "Give me one synthetic customer review."}],
    )
    print(resp.choices[0].message.content)
# Server is torn down automatically on exit.
