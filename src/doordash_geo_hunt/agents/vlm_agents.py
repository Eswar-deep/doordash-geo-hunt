from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ..llm_vision import VisionTask, active_vision_provider, vision_prompt
from ..models import AgentName, AgentResult, ContestInput, LocationCandidate, SearchRegion


def _extract_candidates(text: str, agent: AgentName) -> list[LocationCandidate]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    rows = json.loads(match.group())
    candidates: list[LocationCandidate] = []
    for row in rows:
        candidates.append(
            LocationCandidate(
                lat=float(row["lat"]),
                lng=float(row["lng"]),
                confidence=float(row.get("confidence", 0.5)),
                agent=agent,
                evidence=row.get("evidence", ""),
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

Return ONLY a JSON array:
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

Return ONLY a JSON array (max 5):
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
                "No vision LLM configured. Set GEMINI_API_KEY (free) or another provider in .env."
            )

        prompt = _vision_prompt(region, mode, ocr_hint)
        task = VisionTask.GEOGUESSER if mode == "geoguesser" else VisionTask.LANDMARK_OCR
        text = vision_prompt(prompt, contest.location_image, task=task)
        candidates = _extract_candidates(text, agent)
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


def run_vlm_geoguesser(contest: ContestInput, region: SearchRegion) -> AgentResult:
    return _run_vlm_agent(contest, region, AgentName.VLM_GEOGUESSER, "geoguesser")


def run_landmark_ocr(contest: ContestInput, region: SearchRegion) -> AgentResult:
    started = time.time()
    ocr_hint = ""
    try:
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        texts = reader.readtext(str(contest.location_image), detail=0)
        if texts:
            ocr_hint = ", ".join(texts[:12])
    except Exception:
        ocr_hint = ""

    result = _run_vlm_agent(contest, region, AgentName.LANDMARK_OCR, "landmark", ocr_hint=ocr_hint)
    if ocr_hint:
        result.notes = f"OCR: {ocr_hint}\n" + result.notes
    result.runtime_s = time.time() - started
    return result
