import os
import httpx
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
region = os.getenv("AWS_BEDROCK_REGION", "us-east-1")
models = [
    "us.anthropic.claude-opus-4-6",
    "anthropic.claude-opus-4-6",
    "us.anthropic.claude-opus-4-6-v1:0",
    "us.anthropic.claude-opus-4-6-20250205-v1:0",
    "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "us.anthropic.claude-sonnet-4-6",
]
for model in models:
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse"
    r = httpx.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"messages": [{"role": "user", "content": [{"text": "OK"}]}]},
        timeout=45,
    )
    snippet = r.text[:100].replace("\n", " ")
    print(f"{model}: {r.status_code} {snippet}")
