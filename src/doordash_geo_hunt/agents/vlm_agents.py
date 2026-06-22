from __future__ import annotations

import json
import re
import threading
import time

from ..llm_vision import VisionTask, active_vision_provider, vision_prompt
from ..models import AgentName, AgentResult, ContestInput, LocationCandidate, SearchRegion
from ..pipeline_context import PipelineContext

# Promo / instructional words baked into DoorDash drop graphics — never a real place.
PROMO_BLOCKLIST = {
    "THE", "BAG", "GRAB", "CLUE", "GO", "FIND", "THEM",
    "DOORDASH", "FIFA", "SEAT", "DROP", "TICKET", "FREE", "WIN",
}

_OCR_LOCK = threading.Lock()
_OCR_READER = None


def get_ocr_reader():
    """Process-wide EasyOCR reader (heavy to construct; warm once)."""
    global _OCR_READER
    with _OCR_LOCK:
        if _OCR_READER is None:
            import easyocr

            _OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _OCR_READER


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences so the JSON regex matches cleanly."""
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return fenced.group(1) if fenced else text


def _filter_ocr_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        clean = tok.strip()
        if len(clean) < 3:
            continue
        if clean.upper() in PROMO_BLOCKLIST:
            continue
        out.append(clean)
    return out


def _extract_candidates(text: str, agent: AgentName, region: SearchRegion) -> list[LocationCandidate]:
    body = _strip_code_fences(text)
    match = re.search(r"\[[\s\S]*\]", body)
    if not match:
        return []
    try:
        rows = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    candidates: list[LocationCandidate] = []
    for row in rows:
        try:
            lat, lng = float(row["lat"]), float(row["lng"])
        except (KeyError, TypeError, ValueError):
            continue
        # Reject hallucinated coordinates outside the search circle.
        if not region.contains(lat, lng):
            continue
        candidates.append(
            LocationCandidate(
                lat=lat,
                lng=lng,
                confidence=float(row.get("confidence", 0.5)),
                agent=agent,
                evidence=str(row.get("evidence", "")),
            )
        )
    return sorted(candidates, key=lambda c: c.confidence, reverse=True)[:5]


def _vision_prompt(region: SearchRegion, mode: str, ocr_hint: str = "") -> str:
    bounds = (
        f"center=({region.center_lat:.6f}, {region.center_lng:.6f}), "
        f"radius={region.radius_m:.0f}m, city={region.city!r}"
    )
    ocr_block = f"\nLocal OCR hints: {ocr_hint}\n" if ocr_hint else ""
    if mode == "geoguesser":
        return f"""You are Agent: VLM Geoguesser for a DoorDash FIFA ticket drop.

Search constraint (MUST obey): {bounds}
{ocr_block}
Task:
1. Analyze the location photo background (ignore foreground bag/pedestal).
2. Identify architecture, storefronts, street furniture, vegetation, skyline cues.
3. Propose up to 5 lat/lng points INSIDE the circle that best match the scene.

Return ONLY a JSON array (no markdown fences):
[
  {{"lat": float, "lng": float, "confidence": 0-1, "evidence": "why this spot"}}
]
"""
    return f"""You are Agent: Landmark + OCR for a DoorDash FIFA ticket drop.

Search constraint (MUST obey): {bounds}
{ocr_block}
Task:
1. OCR any visible text/signs in the location photo background.
2. Match text + visual landmarks to real places inside the circle.
3. Use Google Maps knowledge for POI names, cross streets, parks, plazas.

Return ONLY a JSON array (max 5, no markdown fences):
[
  {{"lat": float, "lng": float, "confidence": 0-1, "evidence": "sign/landmark match"}}
]
"""


def _run_vlm_agent(
    contest: ContestInput,
    region: SearchRegion,
    agent: AgentName,
    mode: str,
    ocr_hint: str = "",
) -> AgentResult:
    started = time.time()
    try:
        if not active_vision_provider():
            raise RuntimeError(
                "No vision LLM configured. Set VISION_LLM_PROVIDER + keys in .env."
            )

        prompt = _vision_prompt(region, mode, ocr_hint)
        task = VisionTask.GEOGUESSER if mode == "geoguesser" else VisionTask.LANDMARK_OCR
        text = vision_prompt(prompt, contest.location_image, task=task)
        candidates = _extract_candidates(text, agent, region)
        return AgentResult(
            agent=agent,
            candidates=candidates,
            notes=text[:500],
            runtime_s=time.time() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            agent=agent,
            candidates=[],
            error=str(exc),
            runtime_s=time.time() - started,
        )


def run_vlm_geoguesser(ctx: PipelineContext) -> AgentResult:
    return _run_vlm_agent(ctx.contest, ctx.region, AgentName.VLM_GEOGUESSER, "geoguesser")


def run_landmark_ocr(ctx: PipelineContext) -> AgentResult:
    started = time.time()
    ocr_hint = ""
    try:
        reader = get_ocr_reader()
        texts = reader.readtext(str(ctx.contest.location_image), detail=0)
        ocr_hint = ", ".join(_filter_ocr_tokens(texts)[:12])
    except Exception:  # noqa: BLE001
        ocr_hint = ""

    result = _run_vlm_agent(
        ctx.contest, ctx.region, AgentName.LANDMARK_OCR, "landmark", ocr_hint=ocr_hint
    )
    if ocr_hint:
        result.notes = f"OCR: {ocr_hint}\n" + result.notes
    result.runtime_s = time.time() - started
    return result
