from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agents.visual_matcher import run_kartaview_matcher, run_mapillary_matcher, run_streetview_matcher
from .agents.vlm_agents import run_landmark_ocr, run_vlm_geoguesser
from .judge.judge_agent import judge_results
from .map_extractor import extract_region_from_map
from .models import AgentResult, ContestInput, FinalVerdict, SearchRegion


def resolve_region(contest: ContestInput) -> SearchRegion:
    if contest.region_override:
        return contest.region_override
    return extract_region_from_map(contest.map_image, city_hint=contest.city_hint)


def _warmup_clip() -> None:
    """Initialize the shared CLIP model (and its heavy torch/torchvision/open_clip
    imports) on the main thread before the parallel agents start.

    Loading these concurrently from multiple worker threads triggers
    partially-initialized-module / circular-import errors.
    """
    try:
        from .matching.clip_matcher import get_clip_matcher

        get_clip_matcher()
    except Exception:  # noqa: BLE001 - individual agents will surface the error
        pass


def _run_all_agents_sync(contest: ContestInput, region: SearchRegion, cache_dir: Path) -> list[AgentResult]:
    _warmup_clip()
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [
            pool.submit(run_streetview_matcher, contest, region, cache_dir),
            pool.submit(run_mapillary_matcher, contest, region, cache_dir),
            pool.submit(run_kartaview_matcher, contest, region, cache_dir),
            pool.submit(run_vlm_geoguesser, contest, region),
            pool.submit(run_landmark_ocr, contest, region),
        ]
        return [f.result() for f in futures]


async def run_pipeline(
    contest: ContestInput,
    cache_dir: Path | None = None,
) -> tuple[SearchRegion, list[AgentResult], FinalVerdict]:
    contest.validate()
    cache_dir = cache_dir or Path(".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    region = resolve_region(contest)
    loop = asyncio.get_running_loop()
    agent_results = await loop.run_in_executor(None, _run_all_agents_sync, contest, region, cache_dir)
    verdict = judge_results(contest, region, agent_results)
    return region, agent_results, verdict


def run_pipeline_sync(
    contest: ContestInput,
    cache_dir: Path | None = None,
) -> tuple[SearchRegion, list[AgentResult], FinalVerdict]:
    return asyncio.run(run_pipeline(contest, cache_dir=cache_dir))


def format_report(
    region: SearchRegion,
    agent_results: list[AgentResult],
    verdict: FinalVerdict,
) -> str:
    lines = [
        "=== DoorDash Geo Hunt Report ===",
        f"Region: center=({region.center_lat:.6f}, {region.center_lng:.6f}) radius={region.radius_m:.0f}m city={region.city}",
        "",
        "-- Agent outputs --",
    ]
    for result in agent_results:
        lines.append(f"[{result.agent.value}] runtime={result.runtime_s:.1f}s error={result.error}")
        for cand in result.candidates[:3]:
            lines.append(
                f"  - ({cand.lat:.6f}, {cand.lng:.6f}) conf={cand.confidence:.3f} :: {cand.evidence[:100]}"
            )
    lines.extend(
        [
            "",
            "-- Final verdict --",
            f"Location: ({verdict.lat:.6f}, {verdict.lng:.6f})",
            f"Confidence: {verdict.confidence:.3f}",
            f"Winner agent: {verdict.winner_agent.value if verdict.winner_agent else 'ensemble'}",
            f"Reasoning: {verdict.reasoning}",
        ]
    )
    if verdict.street_view_url:
        lines.append(f"Street View: {verdict.street_view_url}")
    return "\n".join(lines)


def save_json_output(
    path: Path,
    region: SearchRegion,
    agent_results: list[AgentResult],
    verdict: FinalVerdict,
) -> None:
    payload = {
        "generated_at": time.time(),
        "region": {
            "center_lat": region.center_lat,
            "center_lng": region.center_lng,
            "radius_m": region.radius_m,
            "city": region.city,
            "source": region.source,
        },
        "agents": [
            {
                "agent": r.agent.value,
                "runtime_s": r.runtime_s,
                "error": r.error,
                "notes": r.notes,
                "candidates": [c.to_dict() for c in r.candidates],
            }
            for r in agent_results
        ],
        "verdict": {
            "lat": verdict.lat,
            "lng": verdict.lng,
            "confidence": verdict.confidence,
            "reasoning": verdict.reasoning,
            "winner_agent": verdict.winner_agent.value if verdict.winner_agent else None,
            "street_view_url": verdict.street_view_url,
            "all_candidates": [c.to_dict() for c in verdict.all_candidates],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
