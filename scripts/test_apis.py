"""Smoke-test API keys from .env (prints status only, never secrets)."""

from __future__ import annotations

import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()


def test(name: str, fn) -> None:
    try:
        ok, msg = fn()
        status = "OK" if ok else "FAIL"
        print(f"{name}: {status} - {msg[:160]}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: FAIL - {exc}")


def main() -> int:
    def t_google():
        key = os.getenv("GOOGLE_MAPS_API_KEY")
        r = httpx.get(
            "https://maps.googleapis.com/maps/api/streetview/metadata",
            params={"location": "25.8117,-80.1932", "key": key},
            timeout=20,
        )
        j = r.json()
        return j.get("status") == "OK", str(j.get("status", r.text[:80]))

    def t_mapillary():
        tok = os.getenv("MAPILLARY_ACCESS_TOKEN")
        r = httpx.get(
            "https://graph.mapillary.com/images",
            params={
                "bbox": "-80.2,25.8,-80.19,25.82",
                "limit": 1,
                "fields": "id",
                "access_token": tok,
            },
            timeout=20,
        )
        return r.status_code == 200, f"status={r.status_code}"

    def t_gemini():
        key = os.getenv("GEMINI_API_KEY")
        last = ""
        for model in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            r = httpx.post(
                url,
                params={"key": key},
                json={"contents": [{"parts": [{"text": "Reply with OK"}]}]},
                timeout=30,
            )
            last = f"{model} status={r.status_code} {r.text[:120]}"
            if r.status_code == 200:
                return True, model
        return False, last

    def t_azure_maps():
        key = os.getenv("AZURE_MAPS_API_KEY")
        r = httpx.get(
            "https://atlas.microsoft.com/search/address/json",
            params={"api-version": "1.0", "query": "Miami FL", "subscription-key": key},
            timeout=20,
        )
        return r.status_code == 200, f"status={r.status_code} {r.text[:80]}"

    def t_bedrock():
        token = (
            os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            or os.getenv("ANTHROPIC_AWS_BEDROCK_API_KEY")
            or os.getenv("ANTHROPIC_AWS_BEDDROCK_API_KEY")
        )
        if not token and not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
            return False, "set AWS_BEARER_TOKEN_BEDROCK or IAM credentials"
        region = os.getenv("AWS_BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))
        sonnet = os.getenv("AWS_BEDROCK_SONNET_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
        opus = os.getenv("AWS_BEDROCK_OPUS_MODEL_ID", "us.anthropic.claude-opus-4-6")
        opus_fb = os.getenv(
            "AWS_BEDROCK_OPUS_FALLBACK_MODEL_ID",
            "us.anthropic.claude-opus-4-5-20251101-v1:0",
        )
        for label, model in (("Sonnet", sonnet), ("Opus", opus), ("Opus-fallback", opus_fb)):
            url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse"
            r = httpx.post(
                url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"messages": [{"role": "user", "content": [{"text": "Reply OK"}]}]},
                timeout=30,
            )
            if r.status_code != 200:
                if label == "Opus" and model != opus_fb:
                    continue
                return False, f"{label} ({model}) status={r.status_code} {r.text[:80]}"
        return True, f"map/OCR={sonnet}, geoguesser/judge={opus} (fallback {opus_fb})"

    def t_anthropic():
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return False, "not set (direct api.anthropic.com only)"
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=30,
        )
        return r.status_code == 200, f"status={r.status_code} {r.text[:100]}"

    def t_azure_openai():
        key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        sonnet = os.getenv("AZURE_OPENAI_DEPLOYMENT_SONNET", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
        opus = os.getenv("AZURE_OPENAI_DEPLOYMENT_OPUS", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
        if not endpoint or not sonnet or not opus:
            return False, "needs AZURE_OPENAI_ENDPOINT + DEPLOYMENT_SONNET + DEPLOYMENT_OPUS"
        for label, deployment in (("Sonnet", sonnet), ("Opus", opus)):
            url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
            r = httpx.post(
                url,
                params={"api-version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")},
                headers={"api-key": key, "content-type": "application/json"},
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
                timeout=30,
            )
            if r.status_code != 200:
                return False, f"{label} ({deployment}) status={r.status_code} {r.text[:80]}"
        return True, f"map/OCR={sonnet}, geoguesser/judge={opus}"

    test("Google Street View", t_google)
    test("Mapillary", t_mapillary)
    test("Gemini", t_gemini)
    test("AWS Bedrock", t_bedrock)
    test("Azure Maps", t_azure_maps)
    test("Anthropic (direct)", t_anthropic)
    test("Azure OpenAI", t_azure_openai)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
