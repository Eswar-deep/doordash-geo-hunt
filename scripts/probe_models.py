"""Probe Bedrock/Gemini vision models (no secrets printed)."""
from __future__ import annotations

import base64
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()


def bedrock_converse(model: str, content: list, *, vision: bool = False) -> tuple[bool, str]:
    token = (
        os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        or os.getenv("ANTHROPIC_AWS_BEDROCK_API_KEY")
    )
    region = os.getenv("AWS_BEDROCK_REGION", "us-east-1")
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse"
    r = httpx.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"messages": [{"role": "user", "content": content}]},
        timeout=90,
    )
    if r.status_code != 200:
        return False, f"{r.status_code} {r.text[:140]}"
    parts = r.json().get("output", {}).get("message", {}).get("content", [])
    text = "".join(p.get("text", "") for p in parts if "text" in p)
    label = "vision" if vision else "text"
    return True, f"[{label}] {text[:160]}"


def main() -> int:
    text_models = [
        "us.anthropic.claude-opus-4-8",
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "us.anthropic.claude-sonnet-4-6",
        "us.amazon.nova-pro-v1:0",
    ]

    print("=== Bedrock TEXT ===")
    for model in text_models:
        ok, msg = bedrock_converse(model, [{"text": "Reply with exactly: OK"}])
        print(f"{'OK' if ok else 'FAIL':4} {model}\n     {msg}")

    gkey = os.getenv("GOOGLE_MAPS_API_KEY")
    if not gkey:
        print("\nSkip vision: no GOOGLE_MAPS_API_KEY")
        return 0

    img = httpx.get(
        "https://maps.googleapis.com/maps/api/streetview",
        params={"size": "400x300", "location": "25.8117,-80.1932", "key": gkey},
        timeout=30,
    )
    img.raise_for_status()
    img_b64 = base64.b64encode(img.content).decode("ascii")
    prompt = 'Describe this scene briefly. Return JSON: {"description": "..."}'

    print("\n=== Bedrock VISION ===")
    vision_content = [
        {"image": {"format": "jpeg", "source": {"bytes": img_b64}}},
        {"text": prompt},
    ]
    for model in text_models:
        ok, msg = bedrock_converse(model, vision_content, vision=True)
        print(f"{'OK' if ok else 'FAIL':4} {model}\n     {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
