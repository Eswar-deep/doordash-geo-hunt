from __future__ import annotations

import base64
import json
import os
import re
from ..geo import haversine_m
from ..llm_vision import VisionTask, active_vision_provider, vision_prompt
from ..models import AgentName, AgentResult, ContestInput, FinalVerdict, LocationCandidate, SearchRegion
from ..streetview import StreetViewClient


def _in_region(region: SearchRegion, lat: float, lng: float) -> bool:
    return region.contains(lat, lng)


def _cluster_votes(candidates: list[LocationCandidate], radius_m: float = 40.0) -> list[LocationCandidate]:
    """Merge cross-agent duplicates and boost confidence when agents agree."""
    if not candidates:
        return []

    clusters: list[list[LocationCandidate]] = []
    for cand in sorted(candidates, key=lambda c: c.confidence, reverse=True):
        placed = False
        for cluster in clusters:
            if haversine_m(cand.lat, cand.lng, cluster[0].lat, cluster[0].lng) <= radius_m:
                cluster.append(cand)
                placed = True
                break
        if not placed:
            clusters.append([cand])

    merged: list[LocationCandidate] = []
    for cluster in clusters:
        agents = {c.agent for c in cluster}
        total_conf = sum(c.confidence for c in cluster)
        agreement_boost = 1.0 + 0.15 * (len(agents) - 1)
        lat = sum(c.lat * c.confidence for c in cluster) / total_conf
        lng = sum(c.lng * c.confidence for c in cluster) / total_conf
        score = min(0.99, (total_conf / len(cluster)) * agreement_boost)
        evidence = " | ".join(f"{c.agent.value}: {c.evidence[:80]}" for c in cluster[:3])
        merged.append(
            LocationCandidate(
                lat=lat,
                lng=lng,
                confidence=score,
                agent=next(iter(agents)),
                evidence=evidence,
                metadata={"agents": [a.value for a in agents], "votes": len(cluster)},
            )
        )
    return sorted(merged, key=lambda c: c.confidence, reverse=True)


def _fetch_verification_panels(
    contest: ContestInput,
    top: list[LocationCandidate],
) -> list[dict]:
    """Pull Street View at candidate headings for side-by-side judge review."""
    if not os.getenv("GOOGLE_MAPS_API_KEY"):
        return [{"candidate": c.to_dict(), "street_view_b64": None} for c in top]

    client = StreetViewClient()
    panels: list[dict] = []
    for cand in top:
        headings = [cand.heading] if cand.heading is not None else [0, 90, 180, 270]
        best_b64 = None
        for heading in headings:
            if heading is None:
                continue
            img = client.fetch_image(cand.lat, cand.lng, heading=float(heading))
            if img is not None:
                from io import BytesIO

                buf = BytesIO()
                img.save(buf, format="JPEG", quality=85)
                best_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                break
        panels.append({"candidate": cand.to_dict(), "street_view_b64": best_b64})
    return panels


def _judge_with_llm(
    contest: ContestInput,
    region: SearchRegion,
    agent_results: list[AgentResult],
    clustered: list[LocationCandidate],
) -> FinalVerdict:
    top = clustered[:5]
    panels = _fetch_verification_panels(contest, top)
    # Do not embed Street View base64 in the text prompt — exceeds model context limits.
    summary_panels = [
        {
            "candidate": p["candidate"],
            "street_view_available": p["street_view_b64"] is not None,
        }
        for p in panels
    ]

    summary = {
        "region": {
            "center_lat": region.center_lat,
            "center_lng": region.center_lng,
            "radius_m": region.radius_m,
            "city": region.city,
        },
        "agent_summaries": [
            {
                "agent": r.agent.value,
                "error": r.error,
                "top": [c.to_dict() for c in r.candidates[:3]],
            }
            for r in agent_results
        ],
        "merged_candidates": [c.to_dict() for c in top],
        "verification_panels": summary_panels,
    }

    prompt = f"""You are the JUDGE agent for a geolocation contest.

Hard rules:
1. Final lat/lng MUST be inside the search circle.
2. Prefer candidates supported by multiple agents.
3. Use agent summaries and merged candidates; Street View was fetched but frames are not attached (see street_view_available flags).
4. Ignore foreground bag/pedestal in the location photo.

Input JSON:
{json.dumps(summary, indent=2)}

Return ONLY JSON:
{{
  "lat": float,
  "lng": float,
  "confidence": 0-1,
  "winner_agent": "streetview_matcher|mapillary_matcher|kartaview_matcher|vlm_geoguesser|landmark_ocr|null",
  "reasoning": "short explanation"
}}
"""

    if active_vision_provider():
        text = vision_prompt(prompt, contest.location_image, task=VisionTask.JUDGE)
    else:
        best = top[0]
        return FinalVerdict(
            lat=best.lat,
            lng=best.lng,
            confidence=best.confidence,
            reasoning="No judge LLM configured; using highest-confidence cluster vote.",
            winner_agent=best.agent,
            all_candidates=clustered,
        )

    data = _extract_verdict_json(text)
    winner = data.get("winner_agent")
    winner_enum = AgentName(winner) if winner in AgentName._value2member_map_ else None

    lat, lng = float(data["lat"]), float(data["lng"])
    if not _in_region(region, lat, lng):
        lat, lng = top[0].lat, top[0].lng

    sv_url = None
    if os.getenv("GOOGLE_MAPS_API_KEY"):
        sv_url = (
            "https://www.google.com/maps/@?api=1&map_action=pano"
            f"&viewpoint={lat},{lng}"
        )

    return FinalVerdict(
        lat=lat,
        lng=lng,
        confidence=float(data.get("confidence", top[0].confidence)),
        reasoning=str(data.get("reasoning", "")),
        winner_agent=winner_enum,
        all_candidates=clustered,
        street_view_url=sv_url,
    )


def _extract_verdict_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"Judge returned no JSON: {text[:400]}")
    return json.loads(match.group())


def judge_results(
    contest: ContestInput,
    region: SearchRegion,
    agent_results: list[AgentResult],
) -> FinalVerdict:
    valid: list[LocationCandidate] = []
    for result in agent_results:
        for cand in result.candidates:
            if _in_region(region, cand.lat, cand.lng):
                valid.append(cand)

    clustered = _cluster_votes(valid)
    if not clustered:
        raise RuntimeError("No in-circle candidates from any agent.")

    return _judge_with_llm(contest, region, agent_results, clustered)
