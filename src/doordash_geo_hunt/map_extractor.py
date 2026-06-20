from __future__ import annotations

import json
import re
from pathlib import Path

from .llm_vision import VisionTask, active_vision_provider, vision_prompt
from .models import SearchRegion


def _extract_json_blob(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:300]}")
    return json.loads(match.group())


def extract_region_from_map(
    map_image: Path,
    city_hint: str | None = None,
) -> SearchRegion:
    """
    Use a vision LLM to read the circular map overlay.
    Uses GEMINI_API_KEY first if set, then other configured providers.
    """
    if not active_vision_provider():
        raise RuntimeError(
            "Set GEMINI_API_KEY (recommended free tier) or another vision LLM key, "
            "or pass --center-lat/--center-lng/--radius-m manually."
        )

    prompt = f"""You are reading a DoorDash contest map screenshot.
The map shows a circular highlighted region where tickets are hidden.

Return ONLY JSON with:
{{
  "center_lat": float,
  "center_lng": float,
  "radius_m": float,
  "city": string or null
}}

Rules:
- Estimate center from the circle centroid on the map.
- radius_m is the circle radius in meters (typical contest drops: 200-1500m).
- Use visible map labels/landmarks for georeferencing.
- city hint from user: {city_hint!r}
"""

    text = vision_prompt(prompt, map_image, task=VisionTask.MAP)
    data = _extract_json_blob(text)
    provider = active_vision_provider() or "unknown"
    return SearchRegion(
        center_lat=float(data["center_lat"]),
        center_lng=float(data["center_lng"]),
        radius_m=float(data["radius_m"]),
        city=data.get("city") or city_hint,
        source=f"{provider}_vision",
    )
