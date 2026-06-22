from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from .agents.visual_matcher import (
    run_kartaview_matcher,
    run_mapillary_matcher,
    run_streetview_matcher,
)
from .agents.vlm_agents import run_landmark_ocr, run_vlm_geoguesser
from .judge.judge_agent import _cluster_votes, judge_results
from .map_extractor import extract_region_from_map
from .models import AgentName, AgentResult, ContestInput, FinalVerdict, SearchRegion
from .pipeline_context import PipelineContext, StreetViewConfig
from .preprocessing import crop_location_background, enhance_for_matching, load_rgb, resize_max_side

DEFAULT_AGENTS = ["streetview", "vlm"]
ALL_AGENT_TOKENS = ["streetview", "vlm", "landmark", "mapillary", "kartaview"]

TOKEN_TO_AGENT: dict[str, AgentName] = {
    "streetview": AgentName.STREETVIEW_MATCHER,
    "vlm": AgentName.VLM_GEOGUESSER,
    "landmark": AgentName.LANDMARK_OCR,
    "mapillary": AgentName.MAPILLARY_MATCHER,
    "kartaview": AgentName.KARTAVIEW_MATCHER,
}


@dataclass
class AgentTimeouts:
    streetview: float = 900.0
    vlm: float = 90.0
    landmark: float = 120.0
    mapillary: float = 300.0
    kartaview: float = 300.0

    def for_token(self, token: str) -> float:
        return float(getattr(self, token, 300.0))


@dataclass
class PipelineConfig:
    agents: list[str] = field(default_factory=lambda: list(DEFAULT_AGENTS))
    staged: bool = True
    staged_parallel: bool = True
    sv: StreetViewConfig = field(default_factory=StreetViewConfig)
    timeouts: AgentTimeouts = field(default_factory=AgentTimeouts)
    judge_workers: int = 4
    cache_dir: Path | None = None
    run_judge: bool = True


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def resolve_region(contest: ContestInput) -> SearchRegion:
    if contest.region_override:
        return contest.region_override
    return extract_region_from_map(contest.map_image, city_hint=contest.city_hint)


def _preflight(contest: ContestInput, agents: list[str]) -> None:
    """Fail fast (<5s) on missing keys before kicking off a long Street View run."""
    import os

    from .llm_vision import active_vision_provider

    if "streetview" in agents and not os.getenv("GOOGLE_MAPS_API_KEY"):
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY is not set — required for the streetview agent. "
            "Set it or drop streetview from --agents."
        )
    if contest.region_override is None and not active_vision_provider():
        raise RuntimeError(
            "No vision LLM configured (needed to read the map circle). "
            "Set VISION_LLM_PROVIDER + keys, or pass --center-lat/--center-lng/--radius-m."
        )


def _build_query_image(contest: ContestInput) -> Image.Image:
    raw = load_rgb(contest.location_image)
    masked = crop_location_background(raw)
    enhanced = enhance_for_matching(masked)
    # Keep image at 768px max — CLIP preprocess does its own final resize, but
    # starting from a higher-res source preserves fine-grained textures
    # (brick patterns, signage, window details) better than the old 512px cap.
    return Image.fromarray(resize_max_side(enhanced, max_side=768))


def _build_query_crops(full_image: Image.Image) -> list[Image.Image]:
    """Generate auxiliary crops for multi-crop CLIP embedding.

    Multiple overlapping crops capture different background regions that may
    match different Street View headings. The final query vector averages all
    crops, making it more robust than a single full-image embedding.
    """
    w, h = full_image.size
    crops: list[Image.Image] = []

    # Top half — buildings/sky/upper architecture (most discriminative for geolocation)
    crops.append(full_image.crop((0, 0, w, h // 2)))

    # Left third and right third — capture side buildings/walls
    third = w // 3
    crops.append(full_image.crop((0, 0, third + third // 2, h)))
    crops.append(full_image.crop((w - third - third // 2, 0, w, h)))

    # Center strip (avoids masked bag region which is now neutral gray)
    margin = w // 5
    crops.append(full_image.crop((margin, 0, w - margin, int(h * 0.6))))

    return crops


def build_context(
    contest: ContestInput, region: SearchRegion, cfg: PipelineConfig
) -> PipelineContext:
    """Preprocess once and warm heavy singletons on the main thread."""
    query_image = _build_query_image(contest)
    query_crops = _build_query_crops(query_image)

    matcher = None
    try:
        from .matching.clip_matcher import get_clip_matcher

        matcher = get_clip_matcher()
    except Exception as exc:  # noqa: BLE001
        _log(f"[warmup] CLIP load failed: {exc}")

    try:
        import torch

        torch.set_num_threads(4)
    except Exception:  # noqa: BLE001
        pass

    if "landmark" in cfg.agents:
        try:
            from .agents.vlm_agents import get_ocr_reader

            get_ocr_reader()
        except Exception as exc:  # noqa: BLE001
            _log(f"[warmup] EasyOCR load failed: {exc}")

    sv_client = None
    if "streetview" in cfg.agents:
        try:
            from .streetview import StreetViewClient

            sv_client = StreetViewClient(workers=cfg.sv.workers)
        except Exception as exc:  # noqa: BLE001
            _log(f"[warmup] Street View client unavailable: {exc}")

    return PipelineContext(
        contest=contest,
        region=region,
        query_image=query_image,
        query_crops=query_crops,
        clip_matcher=matcher,
        sv_client=sv_client,
    )


def _agent_callable(token: str, ctx: PipelineContext, cfg: PipelineConfig) -> Callable[[], AgentResult]:
    cache = cfg.cache_dir
    if token == "streetview":
        return lambda: run_streetview_matcher(ctx, cfg.sv, cache)
    if token == "vlm":
        return lambda: run_vlm_geoguesser(ctx)
    if token == "landmark":
        return lambda: run_landmark_ocr(ctx)
    if token == "mapillary":
        return lambda: run_mapillary_matcher(ctx, cache)
    if token == "kartaview":
        return lambda: run_kartaview_matcher(ctx, cache)
    raise ValueError(f"Unknown agent token: {token}")


def _empty_result(token: str, note: str) -> AgentResult:
    return AgentResult(agent=TOKEN_TO_AGENT[token], candidates=[], notes=note, error="timeout")


def _interim_verdict(
    region: SearchRegion,
    agent_results: list[AgentResult],
    *,
    stage: str,
    agents_pending: list[str],
) -> FinalVerdict | None:
    valid = [
        c for r in agent_results for c in r.candidates if region.contains(c.lat, c.lng)
    ]
    clustered = _cluster_votes(valid)
    if not clustered:
        return None
    best = clustered[0]
    votes = max((int(c.metadata.get("votes", 1)) for c in clustered), default=1)
    conf = min(best.confidence, 0.75)  # provisional confidence capped
    return FinalVerdict(
        lat=best.lat,
        lng=best.lng,
        confidence=conf,
        reasoning=f"Provisional {stage} verdict (weighted cluster lead).",
        winner_agent=best.agent,
        all_candidates=clustered,
        low_confidence=best.confidence < 0.6,
        human_review=best.confidence < 0.6,
        agreement_votes=votes,
        alternatives=clustered[1:4],
        stage=stage,
        provisional=True,
        agents_pending=agents_pending,
    )


StageCallback = Callable[[str, SearchRegion, FinalVerdict, list[AgentResult]], None]


def run_contest(
    contest: ContestInput,
    cfg: PipelineConfig,
    on_stage: StageCallback | None = None,
) -> tuple[SearchRegion, list[AgentResult], FinalVerdict]:
    contest.validate()
    agents = [a for a in cfg.agents if a in TOKEN_TO_AGENT] or list(DEFAULT_AGENTS)
    enabled = {TOKEN_TO_AGENT[a] for a in agents}

    _preflight(contest, agents)
    region = resolve_region(contest)
    _log(
        f"[region] center=({region.center_lat:.6f},{region.center_lng:.6f}) "
        f"radius={region.radius_m:.0f}m city={region.city} agents={agents}"
    )

    ctx = build_context(contest, region, cfg)
    results: dict[str, AgentResult] = {}
    start = time.time()
    pool = ThreadPoolExecutor(max_workers=max(1, len(agents)))
    futures = {pool.submit(_agent_callable(a, ctx, cfg)): a for a in agents}
    token_future = {tok: fut for fut, tok in futures.items()}

    def _collect(token: str) -> AgentResult:
        if token in results:
            return results[token]
        fut = token_future[token]
        try:
            res = fut.result(timeout=cfg.timeouts.for_token(token))
        except FutureTimeout:
            _log(f"[{token}] TIMEOUT after {cfg.timeouts.for_token(token):.0f}s")
            res = _empty_result(token, "agent timed out")
        results[token] = res
        _log(f"[{token}] done {res.runtime_s:.1f}s candidates={len(res.candidates)} error={res.error}")
        return res

    try:
        if cfg.staged and cfg.staged_parallel:
            _log(f"[stage] parallel start t=0 agents={agents}")
            stage_num = 0
            max_timeout = max(cfg.timeouts.for_token(a) for a in agents)
            for fut in as_completed(futures, timeout=max_timeout):
                token = futures[fut]
                try:
                    res = fut.result(timeout=0)
                except Exception as exc:  # noqa: BLE001
                    _log(f"[{token}] failed: {exc}")
                    res = _empty_result(token, str(exc))
                results[token] = res
                _log(f"[{token}] done {res.runtime_s:.1f}s candidates={len(res.candidates)} error={res.error}")
                stage_num += 1
                pending = [a for a in agents if a not in results]
                got = [results[t] for t in agents if t in results]
                stage_label = f"p{stage_num}"
                v = _interim_verdict(region, got, stage=stage_label, agents_pending=pending)
                if v and on_stage:
                    on_stage(stage_label, region, v, got)
        else:
            for token in agents:
                _collect(token)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    agent_results = [results[a] for a in agents if a in results]
    if not cfg.run_judge:
        _log(f"[stage] agents complete in {time.time() - start:.1f}s; judge skipped (--stage)")
        interim = _interim_verdict(region, agent_results, stage="p3_final", agents_pending=[])
        if interim is None:
            raise RuntimeError("No in-circle candidates from any agent.")
        if ctx.sv_client is not None:
            try:
                ctx.sv_client.close()
            except Exception:  # noqa: BLE001
                pass
        if on_stage:
            on_stage("p3_final", region, interim, agent_results)
        return region, agent_results, interim

    _log(f"[stage] agents complete in {time.time() - start:.1f}s; judging")
    verdict = judge_results(
        contest, region, agent_results, enabled_agents=enabled, judge_workers=cfg.judge_workers
    )
    verdict.stage = "p3_final"
    if ctx.sv_client is not None:
        try:
            ctx.sv_client.close()
        except Exception:  # noqa: BLE001
            pass
    if on_stage:
        on_stage("p3_final", region, verdict, agent_results)
    return region, agent_results, verdict


def run_pipeline_sync(
    contest: ContestInput,
    cache_dir: Path | None = None,
    cfg: PipelineConfig | None = None,
) -> tuple[SearchRegion, list[AgentResult], FinalVerdict]:
    cfg = cfg or PipelineConfig(staged=False, staged_parallel=False)
    if cache_dir is not None:
        cfg.cache_dir = cache_dir
    return run_contest(contest, cfg)


def format_report(
    region: SearchRegion,
    agent_results: list[AgentResult],
    verdict: FinalVerdict,
) -> str:
    lines = [
        "=== DoorDash Geo Hunt Report ===",
        f"Region: center=({region.center_lat:.6f}, {region.center_lng:.6f}) "
        f"radius={region.radius_m:.0f}m city={region.city}",
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
            f"Confidence: {verdict.confidence:.3f}"
            + ("  [LOW CONFIDENCE]" if verdict.low_confidence else "")
            + ("  [HUMAN REVIEW]" if verdict.human_review else ""),
            f"Agent agreement: {verdict.agreement_votes} agent(s) on top cluster",
            f"Winner agent: {verdict.winner_agent.value if verdict.winner_agent else 'ensemble'}",
            f"Reasoning: {verdict.reasoning}",
        ]
    )
    if verdict.human_review and verdict.alternatives:
        lines.append("Alternatives to check:")
        for alt in verdict.alternatives:
            lines.append(
                f"  - ({alt.lat:.6f}, {alt.lng:.6f}) conf={alt.confidence:.3f} :: {alt.evidence[:80]}"
            )
    if verdict.street_view_url:
        lines.append(f"Street View: {verdict.street_view_url}")
    lines.append(f"Google Maps: {verdict.maps_url()}")
    return "\n".join(lines)


def verdict_payload(verdict: FinalVerdict) -> dict:
    return {
        "stage": verdict.stage,
        "provisional": verdict.provisional,
        "agents_pending": verdict.agents_pending,
        "lat": verdict.lat,
        "lng": verdict.lng,
        "confidence": verdict.confidence,
        "low_confidence": verdict.low_confidence,
        "human_review": verdict.human_review,
        "agreement_votes": verdict.agreement_votes,
        "reasoning": verdict.reasoning,
        "winner_agent": verdict.winner_agent.value if verdict.winner_agent else None,
        "street_view_url": verdict.street_view_url,
        "maps_url": verdict.maps_url(),
        "alternatives": [c.to_dict() for c in verdict.alternatives],
        "all_candidates": [c.to_dict() for c in verdict.all_candidates],
    }


def save_json_output(
    path: Path,
    region: SearchRegion,
    agent_results: list[AgentResult],
    verdict: FinalVerdict,
) -> None:
    payload = {
        "generated_at": time.time(),
        "stage": verdict.stage,
        "provisional": verdict.provisional,
        "agents_pending": verdict.agents_pending,
        "human_review": verdict.human_review,
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
        "verdict": verdict_payload(verdict),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
