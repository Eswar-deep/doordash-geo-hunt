from __future__ import annotations

import sys
import time
from pathlib import Path

from ..geo import cluster_candidates, cluster_scored_points
from ..matching.clip_matcher import get_clip_matcher
from ..models import AgentName, AgentResult, LocationCandidate
from ..pipeline_context import PipelineContext, StreetViewConfig
from ..streetview import StreetViewClient, headings_evenly


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def percentile_confidence(score: float, sorted_scores: list[float], lo: float = 0.30, hi: float = 0.95) -> float:
    """Map a raw cosine score to a percentile-based confidence in [lo, hi].

    This deliberately compresses absolute CLIP scores (which cluster tightly and
    over-inflate under the old ``(score+1)/2`` formula) into a within-run ranking
    so the judge can compare CLIP against VLM fairly.
    """
    if not sorted_scores:
        return lo
    n = len(sorted_scores)
    # Count how many scores are <= this score.
    below = sum(1 for s in sorted_scores if s <= score)
    pct = below / n
    return round(lo + (hi - lo) * pct, 4)


def _heading_count(cfg: StreetViewConfig, default: int) -> int:
    if cfg.heading_step:
        return max(1, int(round(360 / cfg.heading_step)))
    if cfg.headings_override:
        return cfg.headings_override
    return default


def _build_tasks(
    panos: list[dict],
    headings: list[float],
    *,
    pitch: float = 0.0,
    cap: int | None = None,
) -> list[dict]:
    tasks: list[dict] = []
    for pano in panos:
        for h in headings:
            tasks.append(
                {
                    "lat": pano["lat"],
                    "lng": pano["lng"],
                    "heading": float(h),
                    "pitch": float(pitch),
                    "pano_id": pano.get("pano_id"),
                }
            )
            if cap is not None and len(tasks) >= cap:
                return tasks
    return tasks


def run_streetview_matcher(
    ctx: PipelineContext,
    cfg: StreetViewConfig | None = None,
    cache_dir: Path | None = None,
) -> AgentResult:
    started = time.time()
    cfg = cfg or StreetViewConfig()
    region = ctx.region
    own_client = False
    client = ctx.sv_client
    try:
        if client is None:
            client = StreetViewClient(workers=cfg.workers)
            own_client = True
        matcher = ctx.clip_matcher or get_clip_matcher()
        query_vec = matcher.embed(ctx.query_image)
        sv_cache = (cache_dir / "streetview") if (cache_dir and cfg.cache) else None

        all_scores: list = []  # MatchScore objects across passes
        frames_fetched = 0

        if cfg.coarse_fine:
            # ---- Pass 1: coarse, wide FOV, full circle -----------------------
            coarse_step = cfg.step_m or max(60.0, region.radius_m / 10.0)
            panos = client.list_panoramas(
                region, step_m=coarse_step, max_panos=600, workers=cfg.workers
            )
            coarse_headings = headings_evenly(cfg.headings_coarse)
            tasks1 = _build_tasks(panos, coarse_headings, cap=cfg.max_frames // 2)
            _log(f"[sv] pass1 panos={len(panos)} frames={len(tasks1)}")
            frames1 = client.fetch_frames(
                tasks1, fov=cfg.fov_coarse, workers=cfg.workers, cache_dir=sv_cache, label="sv pass1"
            )
            frames_fetched += len(tasks1)
            ranked1 = matcher.rank_batched(
                query_vec, frames1, top_k=50, batch_size=cfg.clip_batch_size
            )
            all_scores.extend(ranked1)

            # ---- Pass 2: fine, narrow FOV, around top-5 coarse hits ----------
            top5 = cluster_scored_points(
                [(m.lat, m.lng, m.score, m.heading) for m in ranked1], merge_radius_m=30.0
            )[:5]
            remaining = max(0, cfg.max_frames - frames_fetched)
            if top5 and remaining > 0:
                near = client.panoramas_near(
                    [(p.lat, p.lng) for p in top5],
                    region=region,
                    radius_m=60.0,
                    step_m=cfg.step_m or 25.0,
                    max_panos=200,
                    workers=cfg.workers,
                )
                fine_headings = headings_evenly(cfg.headings_fine)
                tasks2 = _build_tasks(near, fine_headings, cap=remaining)
                _log(f"[sv] pass2 panos={len(near)} frames={len(tasks2)} headings={cfg.headings_fine}")
                frames2 = client.fetch_frames(
                    tasks2, fov=cfg.fov_fine, workers=cfg.workers, cache_dir=sv_cache, label="sv pass2"
                )
                frames_fetched += len(tasks2)
                ranked2 = matcher.rank_batched(
                    query_vec, frames2, top_k=50, batch_size=cfg.clip_batch_size
                )
                all_scores.extend(ranked2)

            # ---- Pass 3: heading + pitch refine on top-3 panos ---------------
            if cfg.refine_headings and all_scores:
                top3 = sorted(all_scores, key=lambda m: m.score, reverse=True)[:3]
                refine_tasks: list[dict] = []
                for m in top3:
                    base = int(m.heading) if m.heading is not None else 0
                    span, step = cfg.refine_span, max(1, cfg.refine_step)
                    refine_headings = list(range(base - span, base + span + 1, step))
                    for h in refine_headings:
                        for pitch in cfg.pitch_refine:
                            refine_tasks.append(
                                {
                                    "lat": m.lat,
                                    "lng": m.lng,
                                    "heading": float(h % 360),
                                    "pitch": float(pitch),
                                    "pano_id": m.source_id or None,
                                }
                            )
                refine_tasks = refine_tasks[: cfg.refine_max_frames]
                if refine_tasks:
                    _log(f"[sv] refine frames={len(refine_tasks)} span={cfg.refine_span} step={cfg.refine_step}")
                    frames3 = client.fetch_frames(
                        refine_tasks, fov=cfg.fov_fine, workers=cfg.workers,
                        cache_dir=sv_cache, label="sv refine",
                    )
                    frames_fetched += len(refine_tasks)
                    ranked3 = matcher.rank_batched(
                        query_vec, frames3, top_k=50, batch_size=cfg.clip_batch_size
                    )
                    all_scores.extend(ranked3)
        else:
            # ---- Single pass over the full circle ----------------------------
            n = _heading_count(cfg, default=8)
            step = cfg.step_m or 40.0
            panos = client.list_panoramas(region, step_m=step, max_panos=600, workers=cfg.workers)
            tasks = _build_tasks(panos, headings_evenly(n), cap=cfg.max_frames)
            _log(f"[sv] single-pass panos={len(panos)} frames={len(tasks)} headings={n}")
            frames = client.fetch_frames(
                tasks, fov=cfg.fov_fine, workers=cfg.workers, cache_dir=sv_cache, label="sv"
            )
            frames_fetched += len(tasks)
            all_scores = matcher.rank_batched(
                query_vec, frames, top_k=80, batch_size=cfg.clip_batch_size
            )

        if not all_scores:
            return AgentResult(
                agent=AgentName.STREETVIEW_MATCHER,
                candidates=[],
                notes=f"No street-view frames matched (frames_fetched={frames_fetched}).",
                runtime_s=time.time() - started,
            )

        clustered = cluster_scored_points(
            [(m.lat, m.lng, m.score, m.heading) for m in all_scores], merge_radius_m=30.0
        )
        sorted_scores = sorted(m.score for m in all_scores)
        top_score = clustered[0].score
        fifth = clustered[4].score if len(clustered) >= 5 else clustered[-1].score
        clip_gap = round(top_score - fifth, 4)

        candidates: list[LocationCandidate] = []
        for pt in clustered[:8]:
            conf = percentile_confidence(pt.score, sorted_scores)
            if clip_gap < 0.02:
                conf = round(conf * 0.85, 4)
            candidates.append(
                LocationCandidate(
                    lat=pt.lat,
                    lng=pt.lng,
                    confidence=conf,
                    agent=AgentName.STREETVIEW_MATCHER,
                    heading=pt.heading,
                    evidence=f"CLIP score={pt.score:.4f} pct_conf={conf:.2f} gap={clip_gap:.3f}",
                    metadata={"clip_score": round(pt.score, 4), "clip_score_gap": clip_gap},
                )
            )

        return AgentResult(
            agent=AgentName.STREETVIEW_MATCHER,
            candidates=candidates,
            notes=(
                f"frames_fetched={frames_fetched} unique_clusters={len(clustered)} "
                f"top_clip={top_score:.3f} gap={clip_gap:.3f}"
            ),
            runtime_s=time.time() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            agent=AgentName.STREETVIEW_MATCHER,
            candidates=[],
            error=str(exc),
            runtime_s=time.time() - started,
        )
    finally:
        if own_client and client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass


def _clip_region_matcher(
    ctx: PipelineContext,
    agent: AgentName,
    samples: list[dict],
    label: str,
) -> AgentResult:
    """Shared CLIP ranking for optional Mapillary / KartaView agents."""
    started = time.time()
    matcher = ctx.clip_matcher or get_clip_matcher()
    query_vec = matcher.embed(ctx.query_image)
    ranked = matcher.rank_batched(query_vec, samples, top_k=20)
    merged = cluster_candidates([(m.lat, m.lng, m.score) for m in ranked], merge_radius_m=30.0)
    sorted_scores = sorted(m.score for m in ranked)
    candidates = [
        LocationCandidate(
            lat=lat,
            lng=lng,
            confidence=percentile_confidence(score, sorted_scores),
            agent=agent,
            evidence=f"CLIP {label} match score={score:.4f}",
            metadata={"clip_score": round(score, 4)},
        )
        for lat, lng, score in merged[:8]
    ]
    return AgentResult(
        agent=agent,
        candidates=candidates,
        notes=f"Compared against {len(samples)} {label} frames.",
        runtime_s=time.time() - started,
    )


def run_kartaview_matcher(ctx: PipelineContext, cache_dir: Path | None = None) -> AgentResult:
    started = time.time()
    try:
        from ..kartaview import KartaViewClient

        client = KartaViewClient()
        kv_cache = (cache_dir / "kartaview") if cache_dir else None
        samples = client.images_in_region(ctx.region, cache_dir=kv_cache)
        return _clip_region_matcher(ctx, AgentName.KARTAVIEW_MATCHER, samples, "KartaView")
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            agent=AgentName.KARTAVIEW_MATCHER,
            candidates=[],
            error=str(exc),
            runtime_s=time.time() - started,
        )


def run_mapillary_matcher(ctx: PipelineContext, cache_dir: Path | None = None) -> AgentResult:
    started = time.time()
    try:
        from ..mapillary import MapillaryClient

        client = MapillaryClient()
        m_cache = (cache_dir / "mapillary") if cache_dir else None
        samples = client.images_in_region(ctx.region, cache_dir=m_cache)
        return _clip_region_matcher(ctx, AgentName.MAPILLARY_MATCHER, samples, "Mapillary")
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            agent=AgentName.MAPILLARY_MATCHER,
            candidates=[],
            error=str(exc),
            runtime_s=time.time() - started,
        )
