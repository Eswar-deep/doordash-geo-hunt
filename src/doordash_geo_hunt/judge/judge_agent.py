from __future__ import annotations

import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..geo import haversine_m
from ..llm_vision import VisionTask, active_vision_provider, vision_prompt_multi
from ..models import AgentName, AgentResult, ContestInput, FinalVerdict, LocationCandidate, SearchRegion

# Calibration knobs (env-overridable).
CLUSTER_RADIUS_M = float(os.getenv("JUDGE_CLUSTER_RADIUS_M", "150"))
DISAGREE_M = float(os.getenv("JUDGE_DISAGREE_M", "200"))
HUMAN_REVIEW_CONF = float(os.getenv("JUDGE_HUMAN_REVIEW_CONF", "0.6"))
LOW_GAP = float(os.getenv("JUDGE_LOW_GAP", "0.02"))

AGENT_WEIGHTS: dict[AgentName, float] = {
    AgentName.VLM_GEOGUESSER: 1.1,
    AgentName.STREETVIEW_MATCHER: 1.0,
    AgentName.LANDMARK_OCR: 1.05,
    AgentName.MAPILLARY_MATCHER: 0.9,
    AgentName.KARTAVIEW_MATCHER: 0.9,
}


def _in_region(region: SearchRegion, lat: float, lng: float) -> bool:
    return region.contains(lat, lng)


def _weighted_conf(cand: LocationCandidate) -> float:
    weight = AGENT_WEIGHTS.get(cand.agent, 1.0)
    # Down-rank a Street View hit whose CLIP scores are nearly flat (ambiguous).
    if cand.agent == AgentName.STREETVIEW_MATCHER:
        if cand.metadata.get("clip_score_gap", 1.0) < LOW_GAP:
            weight *= 0.7
    return cand.confidence * weight


def _cluster_votes(
    candidates: list[LocationCandidate], radius_m: float = CLUSTER_RADIUS_M
) -> list[LocationCandidate]:
    """Merge cross-agent duplicates, weight by agent, boost on agreement."""
    if not candidates:
        return []

    ordered = sorted(candidates, key=_weighted_conf, reverse=True)
    clusters: list[list[LocationCandidate]] = []
    for cand in ordered:
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
        weights = [_weighted_conf(c) for c in cluster]
        total_w = sum(weights) or 1e-9
        agreement_boost = 1.0 + 0.2 * (len(agents) - 1)
        lat = sum(c.lat * w for c, w in zip(cluster, weights)) / total_w
        lng = sum(c.lng * w for c, w in zip(cluster, weights)) / total_w
        score = min(0.99, (total_w / len(cluster)) * agreement_boost)
        best = max(cluster, key=_weighted_conf)
        evidence = " | ".join(f"{c.agent.value}: {c.evidence[:80]}" for c in cluster[:3])
        merged.append(
            LocationCandidate(
                lat=lat,
                lng=lng,
                confidence=score,
                agent=best.agent,
                heading=best.heading,
                pitch=best.pitch,
                evidence=evidence,
                metadata={
                    "agents": [a.value for a in agents],
                    "votes": len(cluster),
                    "clip_score_gap": best.metadata.get("clip_score_gap"),
                },
            )
        )
    return sorted(merged, key=lambda c: c.confidence, reverse=True)


def _save_panel(img, dest: Path) -> Path | None:
    if img is None:
        return None
    w, h = img.size
    scale = 512 / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    img.save(dest, format="JPEG", quality=85)
    return dest


def _fetch_verification_panels(
    top: list[LocationCandidate],
    workdir: Path,
    workers: int = 4,
) -> list[tuple[LocationCandidate, Path | None]]:
    """Fetch Street View panels for the top candidates in parallel."""
    if not os.getenv("GOOGLE_MAPS_API_KEY"):
        return [(c, None) for c in top]

    from ..streetview import StreetViewClient

    client = StreetViewClient(workers=workers)

    def _one(idx_cand: tuple[int, LocationCandidate]) -> tuple[LocationCandidate, Path | None]:
        idx, cand = idx_cand
        headings = [cand.heading] if cand.heading is not None else [0.0, 90.0, 180.0, 270.0]
        for heading in headings:
            if heading is None:
                continue
            img = client.fetch_image(cand.lat, cand.lng, heading=float(heading))
            if img is not None:
                return cand, _save_panel(img, workdir / f"panel_{idx}.jpg")
        return cand, None

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_one, list(enumerate(top))))
    finally:
        client.close()


def _judge_with_llm(
    contest: ContestInput,
    region: SearchRegion,
    agent_results: list[AgentResult],
    clustered: list[LocationCandidate],
    enabled_agents: set[AgentName],
    judge_workers: int = 4,
) -> FinalVerdict:
    top = clustered[:5]
    votes = max((int(c.metadata.get("votes", 1)) for c in clustered), default=1)
    spread_m = haversine_m(top[0].lat, top[0].lng, top[1].lat, top[1].lng) if len(top) >= 2 else 0.0

    summary = {
        "region": {
            "center_lat": region.center_lat,
            "center_lng": region.center_lng,
            "radius_m": region.radius_m,
            "city": region.city,
        },
        "agreement": {"max_agent_votes": votes, "top_candidate_spread_m": round(spread_m, 1)},
        "agent_summaries": [
            {"agent": r.agent.value, "error": r.error, "top": [c.to_dict() for c in r.candidates[:3]]}
            for r in agent_results
        ],
        "merged_candidates": [c.to_dict() for c in top],
    }

    if not active_vision_provider():
        best = top[0]
        low = votes < 2 and spread_m > DISAGREE_M
        return FinalVerdict(
            lat=best.lat,
            lng=best.lng,
            confidence=min(best.confidence, 0.45) if low else best.confidence,
            reasoning="No judge LLM configured; using top weighted cluster vote.",
            winner_agent=best.agent,
            all_candidates=clustered,
            low_confidence=low,
            human_review=low or best.confidence < HUMAN_REVIEW_CONF,
            agreement_votes=votes,
            alternatives=clustered[1:4],
        )

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        panels = _fetch_verification_panels(top[:2], workdir, workers=judge_workers)
        images: list[Path] = [contest.location_image]
        panel_notes = []
        for i, (cand, path) in enumerate(panels):
            if path is not None:
                images.append(path)
                panel_notes.append(f"Image {len(images)}: Street View at candidate {i + 1} "
                                   f"({cand.lat:.5f},{cand.lng:.5f}) heading={cand.heading}")
        n_imgs = len(images)
        prompt = f"""You are the JUDGE for a geolocation contest.

Image 1 is the LOCATION CLUE photo (ignore the foreground bag/pedestal — judge the BACKGROUND only).
{chr(10).join(panel_notes) if panel_notes else "No Street View panels were available."}

Compare the clue background against each Street View panel. Pick the candidate whose
architecture / storefronts / street layout best matches. Final lat/lng MUST be inside the circle.
Prefer candidates supported by multiple agents (agreement.max_agent_votes). If support is weak
(votes==1) and candidates are far apart, set confidence LOW (<=0.45).

Context JSON:
{json.dumps(summary, indent=2)}

Return ONLY JSON:
{{"lat": float, "lng": float, "confidence": 0-1,
  "winner_agent": "streetview_matcher|vlm_geoguesser|landmark_ocr|mapillary_matcher|kartaview_matcher|null",
  "reasoning": "short"}}
"""
        import sys

        print(f"[judge] sent {n_imgs} images to {active_vision_provider()}", file=sys.stderr, flush=True)
        text = vision_prompt_multi(prompt, images, task=VisionTask.JUDGE)

    data = _extract_verdict_json(text)
    winner = data.get("winner_agent")
    winner_enum = AgentName(winner) if winner in AgentName._value2member_map_ else None
    if winner_enum is not None and winner_enum not in enabled_agents:
        winner_enum = None

    lat, lng = float(data["lat"]), float(data["lng"])
    confidence = float(data.get("confidence", top[0].confidence))
    if not _in_region(region, lat, lng):
        lat, lng = top[0].lat, top[0].lng
        confidence = min(confidence, top[0].confidence * 0.9)

    low = votes < 2 and spread_m > DISAGREE_M
    if low:
        confidence = min(confidence, 0.45)
    human_review = low or confidence < HUMAN_REVIEW_CONF or spread_m > DISAGREE_M

    reasoning = str(data.get("reasoning", ""))
    if human_review and "low confidence" not in reasoning.lower():
        reasoning = "LOW CONFIDENCE / verify before acting. " + reasoning

    sv_url = None
    if os.getenv("GOOGLE_MAPS_API_KEY"):
        sv_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lng}"

    return FinalVerdict(
        lat=lat,
        lng=lng,
        confidence=confidence,
        reasoning=reasoning,
        winner_agent=winner_enum,
        all_candidates=clustered,
        street_view_url=sv_url,
        low_confidence=low,
        human_review=human_review,
        agreement_votes=votes,
        alternatives=clustered[1:4],
    )


def _extract_verdict_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    body = fenced.group(1) if fenced else text
    match = re.search(r"\{[\s\S]*\}", body)
    if not match:
        raise ValueError(f"Judge returned no JSON: {text[:400]}")
    return json.loads(match.group())


def judge_results(
    contest: ContestInput,
    region: SearchRegion,
    agent_results: list[AgentResult],
    enabled_agents: set[AgentName] | None = None,
    judge_workers: int = 4,
) -> FinalVerdict:
    if enabled_agents is None:
        enabled_agents = {r.agent for r in agent_results}

    valid: list[LocationCandidate] = []
    for result in agent_results:
        for cand in result.candidates:
            if _in_region(region, cand.lat, cand.lng):
                valid.append(cand)

    clustered = _cluster_votes(valid)
    if not clustered:
        raise RuntimeError("No in-circle candidates from any agent.")

    return _judge_with_llm(
        contest, region, agent_results, clustered, enabled_agents, judge_workers=judge_workers
    )
