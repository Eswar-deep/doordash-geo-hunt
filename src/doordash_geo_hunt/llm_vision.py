from __future__ import annotations

import base64
import json
import os
from enum import Enum
from pathlib import Path

import httpx


class VisionTask(str, Enum):
    """Pipeline step — selects Sonnet vs Opus tier when configured."""

    MAP = "map"
    GEOGUESSER = "geoguesser"
    LANDMARK_OCR = "landmark_ocr"
    JUDGE = "judge"


def _uses_sonnet_tier(task: VisionTask) -> bool:
    return task in (VisionTask.MAP, VisionTask.LANDMARK_OCR)


def _extract_json_blob(text: str) -> dict | list:
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        import re

        match = re.search(pattern, text)
        if match:
            return json.loads(match.group())
    raise ValueError(f"No JSON found in model output: {text[:300]}")


def _bedrock_token() -> str | None:
    for name in (
        "AWS_BEARER_TOKEN_BEDROCK",
        "ANTHROPIC_AWS_BEDROCK_API_KEY",
        "ANTHROPIC_AWS_BEDDROCK_API_KEY",  # common typo
    ):
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _bedrock_iam_configured() -> bool:
    return bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))


def active_vision_provider() -> str | None:
    """Return the configured vision LLM provider (optional override, else first match)."""
    forced = os.getenv("VISION_LLM_PROVIDER", "").strip().lower()
    if forced:
        return forced

    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if _bedrock_token() or _bedrock_iam_configured():
        return "bedrock"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return "azure_openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("CURSOR_API_KEY"):
        return "cursor"
    return None


def vision_prompt(
    prompt: str,
    image_path: Path,
    *,
    task: VisionTask = VisionTask.GEOGUESSER,
    cwd: Path | None = None,
) -> str:
    provider = active_vision_provider()
    if provider == "gemini":
        return _gemini_vision(prompt, image_path)
    if provider == "openai":
        return _openai_vision(prompt, image_path, task=task)
    if provider == "azure_openai":
        return _azure_openai_vision(prompt, image_path, task=task)
    if provider == "bedrock":
        return _bedrock_vision(prompt, image_path, task=task)
    if provider == "anthropic":
        return _anthropic_vision(prompt, image_path, task=task)
    if provider == "cursor":
        return _cursor_vision(prompt, image_path, cwd or image_path.parent)
    raise RuntimeError(
        "No vision LLM configured. Set GEMINI_API_KEY, AWS Bedrock (bearer or IAM), "
        "OPENAI_API_KEY, AZURE_OPENAI_*, ANTHROPIC_API_KEY, or CURSOR_API_KEY."
    )


def vision_prompt_multi(
    prompt: str,
    images: list[Path],
    *,
    task: VisionTask = VisionTask.JUDGE,
    cwd: Path | None = None,
) -> str:
    """Send a prompt with multiple images attached (e.g. clue + Street View panels).

    Falls back to single-image ``vision_prompt`` on the first image for providers
    that do not implement a dedicated multi-image path.
    """
    paths = [p for p in images if p is not None]
    if not paths:
        raise ValueError("vision_prompt_multi requires at least one image")
    provider = active_vision_provider()
    if provider == "bedrock":
        return _bedrock_vision_multi(prompt, paths, task=task)
    if provider == "anthropic":
        return _anthropic_vision_multi(prompt, paths, task=task)
    if provider in ("openai", "azure_openai"):
        return _openai_like_vision_multi(prompt, paths, task=task, azure=provider == "azure_openai")
    if provider == "gemini":
        return _gemini_vision_multi(prompt, paths)
    # Cursor or unknown — degrade to single image.
    return vision_prompt(prompt, paths[0], task=task, cwd=cwd)


def _image_b64(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    mime = "image/jpeg"
    if path.suffix.lower() == ".png":
        mime = "image/png"
    return base64.b64encode(data).decode("ascii"), mime


def _image_bytes(path: Path) -> tuple[bytes, str]:
    data = path.read_bytes()
    fmt = "png" if path.suffix.lower() == ".png" else "jpeg"
    return data, fmt


def _gemini_vision(prompt: str, image_path: Path) -> str:
    key = os.environ["GEMINI_API_KEY"]
    primary = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash")
    fallback = os.getenv(
        "GEMINI_VISION_FALLBACK_MODEL",
        "gemini-2.5-flash,gemini-3-flash-preview",
    )
    fallbacks = [m.strip() for m in fallback.split(",") if m.strip()]
    b64, mime = _image_b64(image_path)
    models: list[str] = []
    if primary:
        models.append(primary)
    for fb in fallbacks:
        if fb not in models:
            models.append(fb)
    last_error: Exception | None = None
    for i, model in enumerate(models):
        try:
            return _gemini_generate(key, model, prompt, b64, mime)
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code in (429, 404) and i < len(models) - 1:
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("Gemini vision failed")


def _gemini_generate(key: str, model: str, prompt: str, b64: str, mime: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": b64}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.1},
    }
    resp = httpx.post(url, params={"key": key}, json=payload, timeout=180)
    resp.raise_for_status()
    parts = resp.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def _bedrock_model_for_task(task: VisionTask) -> str:
    per_task = {
        VisionTask.MAP: os.getenv("AWS_BEDROCK_MODEL_MAP"),
        VisionTask.GEOGUESSER: os.getenv("AWS_BEDROCK_MODEL_GEOGUESSER"),
        VisionTask.LANDMARK_OCR: os.getenv("AWS_BEDROCK_MODEL_OCR"),
        VisionTask.JUDGE: os.getenv("AWS_BEDROCK_MODEL_JUDGE"),
    }
    if per_task[task]:
        return per_task[task]
    sonnet = os.getenv("AWS_BEDROCK_SONNET_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    opus = os.getenv("AWS_BEDROCK_OPUS_MODEL_ID", "us.anthropic.claude-opus-4-6")
    if _uses_sonnet_tier(task):
        return sonnet
    return opus


def _bedrock_models_to_try(task: VisionTask) -> list[str]:
    models = [_bedrock_model_for_task(task)]
    if not _uses_sonnet_tier(task):
        fallback = os.getenv(
            "AWS_BEDROCK_OPUS_FALLBACK_MODEL_ID",
            "us.anthropic.claude-opus-4-5-20251101-v1:0",
        )
        if fallback and fallback not in models:
            models.append(fallback)
    return models


def _azure_deployment_for_task(task: VisionTask) -> str:
    per_task = {
        VisionTask.MAP: os.getenv("AZURE_OPENAI_DEPLOYMENT_MAP"),
        VisionTask.GEOGUESSER: os.getenv("AZURE_OPENAI_DEPLOYMENT_GEOGUESSER"),
        VisionTask.LANDMARK_OCR: os.getenv("AZURE_OPENAI_DEPLOYMENT_OCR"),
        VisionTask.JUDGE: os.getenv("AZURE_OPENAI_DEPLOYMENT_JUDGE"),
    }
    if per_task[task]:
        return per_task[task]
    if _uses_sonnet_tier(task):
        return os.getenv(
            "AZURE_OPENAI_DEPLOYMENT_SONNET",
            os.getenv("AZURE_OPENAI_DEPLOYMENT", "claude-sonnet-4-6"),
        )
    return os.getenv(
        "AZURE_OPENAI_DEPLOYMENT_OPUS",
        os.getenv("AZURE_OPENAI_DEPLOYMENT", "claude-opus-4-6"),
    )


def _anthropic_model_for_task(task: VisionTask) -> str:
    if _uses_sonnet_tier(task):
        return os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-6")
    return os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-6")


def _openai_model_for_task(task: VisionTask) -> str:
    if _uses_sonnet_tier(task):
        return os.getenv("OPENAI_VISION_MODEL_SONNET", os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"))
    return os.getenv("OPENAI_VISION_MODEL_OPUS", os.getenv("OPENAI_VISION_MODEL", "gpt-4.1"))


def _openai_vision(prompt: str, image_path: Path, *, task: VisionTask) -> str:
    b64, _ = _image_b64(image_path)
    payload = {
        "model": _openai_model_for_task(task),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "temperature": 0,
    }
    resp = httpx.post(
        f"{os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _azure_openai_vision(prompt: str, image_path: Path, *, task: VisionTask) -> str:
    b64, _ = _image_b64(image_path)
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    deployment = _azure_deployment_for_task(task)
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": 2000,
        "temperature": 0,
    }
    resp = httpx.post(
        url,
        params={"api-version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")},
        headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"], "content-type": "application/json"},
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _anthropic_vision(prompt: str, image_path: Path, *, task: VisionTask) -> str:
    """Direct Anthropic API (console.anthropic.com key — not AWS Bedrock)."""
    b64, mime = _image_b64(image_path)
    media = "image/jpeg" if mime == "image/jpeg" else "image/png"
    payload = {
        "model": _anthropic_model_for_task(task),
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    parts = resp.json().get("content", [])
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def _bedrock_content_blocks(prompt: str, image_paths: list[Path], *, as_base64: bool) -> list[dict]:
    """Build Converse content blocks: all images first, then the text prompt.

    AWS Bedrock's REST Converse API carries blob fields as **base64 strings** in
    JSON, whereas the boto3 client expects **raw bytes** and base64-encodes them
    itself. ``as_base64`` selects the correct encoding for the transport.
    """
    blocks: list[dict] = []
    for path in image_paths:
        raw, fmt = _image_bytes(path)
        payload = base64.b64encode(raw).decode("ascii") if as_base64 else raw
        blocks.append({"image": {"format": fmt, "source": {"bytes": payload}}})
    blocks.append({"text": prompt})
    return blocks


def _bedrock_vision(prompt: str, image_path: Path, *, task: VisionTask) -> str:
    return _bedrock_vision_multi(prompt, [image_path], task=task)


def _bedrock_vision_multi(prompt: str, image_paths: list[Path], *, task: VisionTask) -> str:
    """Claude on Amazon Bedrock via Converse API (bearer token or IAM)."""
    region = os.getenv("AWS_BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))
    token = _bedrock_token()
    last_error = ""
    for model_id in _bedrock_models_to_try(task):
        if token:
            messages = [
                {"role": "user", "content": _bedrock_content_blocks(prompt, image_paths, as_base64=True)}
            ]
            url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse"
            resp = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"messages": messages},
                timeout=240,
            )
            if resp.status_code >= 400:
                last_error = f"{model_id} ({resp.status_code}): {resp.text[:200]}"
                if resp.status_code in (400, 403, 404):
                    continue
                raise RuntimeError(f"Bedrock converse failed ({resp.status_code}): {resp.text[:300]}")
            return _parse_bedrock_converse(resp.json())

        if _bedrock_iam_configured():
            try:
                messages = [
                    {
                        "role": "user",
                        "content": _bedrock_content_blocks(prompt, image_paths, as_base64=False),
                    }
                ]
                return _bedrock_vision_boto3(messages, model_id, region)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{model_id}: {exc}"
                continue

    raise RuntimeError(
        "Bedrock not configured or all models failed. "
        f"Last error: {last_error or 'set AWS_BEARER_TOKEN_BEDROCK or IAM credentials'}"
    )


def _bedrock_vision_boto3(messages: list, model_id: str, region: str) -> str:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)
    response = client.converse(modelId=model_id, messages=messages)
    return _parse_bedrock_converse(response)


def _anthropic_vision_multi(prompt: str, image_paths: list[Path], *, task: VisionTask) -> str:
    content: list[dict] = []
    for path in image_paths:
        b64, mime = _image_b64(path)
        media = "image/jpeg" if mime == "image/jpeg" else "image/png"
        content.append(
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}}
        )
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": _anthropic_model_for_task(task),
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": content}],
    }
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=240,
    )
    resp.raise_for_status()
    parts = resp.json().get("content", [])
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def _openai_like_vision_multi(
    prompt: str, image_paths: list[Path], *, task: VisionTask, azure: bool
) -> str:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        b64, _ = _image_b64(path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    messages = [{"role": "user", "content": content}]
    if azure:
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        deployment = _azure_deployment_for_task(task)
        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions"
        resp = httpx.post(
            url,
            params={"api-version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")},
            headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"], "content-type": "application/json"},
            json={"messages": messages, "max_tokens": 2000, "temperature": 0},
            timeout=240,
        )
    else:
        resp = httpx.post(
            f"{os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={"model": _openai_model_for_task(task), "messages": messages, "temperature": 0},
            timeout=240,
        )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _gemini_vision_multi(prompt: str, image_paths: list[Path]) -> str:
    key = os.environ["GEMINI_API_KEY"]
    model = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash")
    parts: list[dict] = [{"text": prompt}]
    for path in image_paths:
        b64, mime = _image_b64(path)
        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    resp = httpx.post(
        url,
        params={"key": key},
        json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.1}},
        timeout=240,
    )
    resp.raise_for_status()
    out = resp.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in out)


def _parse_bedrock_converse(payload: dict) -> str:
    output = payload.get("output", {})
    message = output.get("message", {})
    parts = message.get("content", [])
    texts: list[str] = []
    for part in parts:
        if "text" in part:
            texts.append(part["text"])
    if not texts:
        raise ValueError(f"Bedrock returned no text: {json.dumps(payload)[:400]}")
    return "".join(texts)


def _cursor_vision(prompt: str, image_path: Path, cwd: Path) -> str:
    from cursor_sdk import Agent, AgentOptions, LocalAgentOptions

    result = Agent.prompt(
        prompt + f"\nImage path: {image_path}",
        AgentOptions(
            api_key=os.environ["CURSOR_API_KEY"],
            model="composer-2.5",
            local=LocalAgentOptions(cwd=str(cwd)),
        ),
    )
    if result.status == "error":
        raise RuntimeError(result.result)
    return result.result or ""
