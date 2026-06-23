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
    """Build fetch tasks with maximum angular diversity per pano.

    When cap < panos×headings, each pano gets a subset of headings that are
    evenly spread across 360° (not clustered at the start of the list). This
    ensures every pano covers all directions even under a tight frame budget.
    """
    if not panos or not headings:
        return []
    import random
    shuffled = list(panos)
    random.shuffle(shuffled)

    if cap is None or cap >= len(shuffled) * len(headings):
        # No cap pressure — give every pano all headings.
        tasks: list[dict] = []
        for pano in shuffled:
            for h in headings:
                tasks.append({
                    "lat": pano["lat"], "lng": pano["lng"],
                    "heading": float(h), "pitch": float(pitch),
                    "pano_id": pano.get("pano_id"),
                })
        return tasks[:cap] if cap else tasks

    # Cap is binding — distribute headings evenly per pano.
    n_per_pano = max(1, cap // len(shuffled))
    n_headings = len(headings)
    tasks = []
    for i, pano in enumerate(shuffled):
        stride = max(1, n_headings // n_per_pano)
        offset = i % stride
        selected = headings[offset::stride][:n_per_pano]
        for h in selected:
            tasks.append({
                "lat": pano["lat"], "lng": pano["lng"],
                "heading": float(h), "pitch": float(pitch),
                "pano_id": pano.get("pano_id"),
            })
    # Fill remaining budget with extra headings for the first panos.
    if len(tasks) < cap:
        used = {(t["pano_id"], t["heading"]) for t in tasks}
        for pano in shuffled:
            for h in headings:
                if (pano.get("pano_id"), float(h)) not in used:
                    tasks.append({
                        "lat": pano["lat"], "lng": pano["lng"],
                        "heading": float(h), "pitch": float(pitch),
                        "pano_id": pano.get("pano_id"),
                    })
                    if len(tasks) >= cap:
                        return tasks
    return tasks[:cap]


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
        query_vec = matcher.embed_multi_crop(ctx.query_image, ctx.query_crops)
        sv_cache = (cache_dir / "streetview") if (cache_dir and cfg.cache) else None

        all_scores: list = []
        frames_fetched = 0

        # ---- Exhaustive pass: ALL panos × N headings × pitches, parallel ---------
        step = cfg.step_m or max(30.0, region.radius_m / 20.0)
        panos = client.list_panoramas(region, step_m=step, max_panos=cfg.max_panos, workers=cfg.workers)
        n_headings = _heading_count(cfg, default=cfg.headings)
        hdgs = headings_evenly(n_headings)

        pitches = getattr(cfg, 'pitch_sweep', (0.0,))
        frames_per_pitch = cfg.max_frames // len(pitches)
        all_tasks: list[dict] = []
        for pitch in pitches:
            tasks_p = _build_tasks(panos, hdgs, pitch=pitch, cap=frames_per_pitch)
            all_tasks.extend(tasks_p)

        _log(f"[sv] exhaustive panos={len(panos)} headings={n_headings} pitches={list(pitches)} frames={len(all_tasks)}")
        frames = client.fetch_frames(
            all_tasks, fov=cfg.fov_fine, workers=cfg.workers, cache_dir=sv_cache, label="sv"
        )
        frames_fetched += len(all_tasks)
        all_scores = list(matcher.rank_batched(
            query_vec, frames, top_k=80, batch_size=cfg.clip_batch_size
        ))

        # ---- Refine: top-1 pano with heading ± span + pitch -------------------
        if cfg.refine_headings and all_scores:
            top1 = all_scores[0]
            base = int(top1.heading) if top1.heading is not None else 0
            span, rstep = cfg.refine_span, max(1, cfg.refine_step)
            refine_tasks: list[dict] = []
            for h in range(base - span, base + span + 1, rstep):
                for pitch in cfg.pitch_refine:
                    refine_tasks.append(
                        {
                            "lat": top1.lat,
                            "lng": top1.lng,
                            "heading": float(h % 360),
                            "pitch": float(pitch),
                            "pano_id": top1.source_id or None,
                        }
                    )
            refine_tasks = refine_tasks[: cfg.refine_max_frames]
            if refine_tasks:
                _log(f"[sv] refine top-1 frames={len(refine_tasks)} span={span} step={rstep}")
                frames_r = client.fetch_frames(
                    refine_tasks, fov=cfg.fov_fine, workers=cfg.workers,
                    cache_dir=sv_cache, label="sv refine",
                )
                frames_fetched += len(refine_tasks)
                ranked_r = matcher.rank_batched(
                    query_vec, frames_r, top_k=10, batch_size=cfg.clip_batch_size
                )
                all_scores.extend(ranked_r)

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
    query_vec = matcher.embed_multi_crop(ctx.query_image, ctx.query_crops)
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


def run_vlm_guided_densification(
    ctx: PipelineContext,
    vlm_candidates: list,
    cfg: StreetViewConfig | None = None,
    radius_m: float = 150.0,
) -> AgentResult:
    """Densify Street View sampling around VLM's best guess, then CLIP-rank.

    This exploits VLM's semantic understanding (which gets ~75m accuracy) by
    exhaustively searching ALL panos × ALL headings × ALL pitches in a small
    radius around the VLM's estimate. CLIP then picks the exact best-match
    frame from this focused, complete pool.
    """
    import time as _time
    started = _time.time()
    cfg = cfg or StreetViewConfig()

    if not vlm_candidates:
        return AgentResult(
            agent=AgentName.STREETVIEW_MATCHER,
            candidates=[],
            notes="No VLM candidates for densification.",
            runtime_s=_time.time() - started,
        )

    own_client = False
    client = ctx.sv_client
    try:
        if client is None:
            from ..streetview import StreetViewClient
            client = StreetViewClient(workers=cfg.workers)
            own_client = True
        matcher = ctx.clip_matcher or get_clip_matcher()
        query_vec = matcher.embed_multi_crop(ctx.query_image, ctx.query_crops)

        # Use up to 10 density centers (VLM + broad CLIP combined), deduplicated
        seen: set[tuple[float, float]] = set()
        centers: list[tuple[float, float]] = []
        for c in vlm_candidates:
            key = (round(c.lat, 4), round(c.lng, 4))
            if key not in seen:
                seen.add(key)
                centers.append((c.lat, c.lng))
            if len(centers) >= 10:
                break
        _log(f"[densify] centers={len(centers)} radius={radius_m}m")

        # Find ALL panos within radius of VLM's guesses
        panos = client.panoramas_near(
            centers,
            region=ctx.region,
            radius_m=radius_m,
            step_m=20.0,  # 20m grid for 300m radius
            max_panos=800,
            workers=cfg.workers,
        )
        _log(f"[densify] found {len(panos)} panos near VLM estimate")

        if not panos:
            return AgentResult(
                agent=AgentName.STREETVIEW_MATCHER,
                candidates=[],
                notes="No panos found near VLM estimate.",
                runtime_s=_time.time() - started,
            )

        # Exhaustive heading × pitch coverage per pano.
        # 24 headings (every 15°) × 4 pitches = 96 frames/pano.
        # With 800 panos = 76,800 total; cap at 38,400 (subsample panos if needed).
        from ..streetview import headings_evenly
        hdgs = headings_evenly(24)  # Every 15° — must cover angles like 135°, 150°
        pitches_local = [0.0, 15.0, 30.0, 45.0]  # 4 pitch levels

        all_tasks: list[dict] = []
        for pano in panos:
            for h in hdgs:
                for p in pitches_local:
                    all_tasks.append({
                        "lat": pano["lat"], "lng": pano["lng"],
                        "heading": float(h), "pitch": float(p),
                        "pano_id": pano.get("pano_id"),
                    })

        # Cap total frames — with 800 panos × 96 = 76800, cap at 38400
        if len(all_tasks) > 38400:
            import random
            random.shuffle(all_tasks)
            all_tasks = all_tasks[:38400]

        _log(f"[densify] fetching {len(all_tasks)} frames ({len(panos)} panos × {len(hdgs)} headings × {len(pitches_local)} pitches)")
        frames = client.fetch_frames(
            all_tasks, fov=90, workers=cfg.workers, label="densify"
        )

        if not frames:
            return AgentResult(
                agent=AgentName.STREETVIEW_MATCHER,
                candidates=[],
                notes="All densify frames failed to download.",
                runtime_s=_time.time() - started,
            )

        # CLIP rank — get top 200, keep ALL images for LoFTR re-ranking
        scored = list(matcher.rank_batched(
            query_vec, frames, top_k=200, batch_size=cfg.clip_batch_size, keep_images=200
        ))

        if not scored:
            return AgentResult(
                agent=AgentName.STREETVIEW_MATCHER,
                candidates=[],
                notes="No densify matches.",
                runtime_s=_time.time() - started,
            )

        clip_top_score = scored[0].score if scored else 0.0
        _log(f"[densify] CLIP top-200 done, top={clip_top_score:.4f}. Running LoFTR...")

        # LoFTR re-ranking (zoom+template now runs earlier in orchestrator)
        from ..matching.feature_matcher import rerank_candidates
        loftr_candidates = [
            {"image": m.image, "lat": m.lat, "lng": m.lng, "heading": m.heading, "score": m.score}
            for m in scored if m.image is not None
        ]

        loftr_discriminative = False
        best_inliers = 0
        if len(loftr_candidates) >= 10:
            loftr_results = rerank_candidates(ctx.query_image, loftr_candidates, top_k=10)
            if loftr_results:
                inlier_counts = [r.num_inliers for _, r in loftr_results[:10]]
                best_inliers = max(inlier_counts)
                worst_top10 = min(inlier_counts)
                loftr_discriminative = (best_inliers - worst_top10) >= 10
                _log(
                    f"[loftr] spread: best={best_inliers} worst_top10={worst_top10} "
                    f"gap={best_inliers - worst_top10} discriminative={loftr_discriminative}"
                )
        else:
            _log(f"[loftr] only {len(loftr_candidates)} candidates with images, skipping")
            loftr_results = []

        # Build retained images lookup
        retained_images: dict[tuple[float, float], "Image.Image"] = {}

        if loftr_discriminative and loftr_results:
            _log(f"[densify] LoFTR discriminative! Using LoFTR ranking.")
            for orig_idx, result in loftr_results[:10]:
                cand = loftr_candidates[orig_idx]
                if cand.get("image") is not None:
                    retained_images[(round(cand["lat"], 6), round(cand["lng"], 6))] = cand["image"]

            for m in scored:
                m.image = None
            for c in loftr_candidates:
                if (round(c["lat"], 6), round(c["lng"], 6)) not in retained_images:
                    c.pop("image", None)

            candidates = []
            for orig_idx, result in loftr_results[:8]:
                cand = loftr_candidates[orig_idx]
                conf = min(result.num_inliers / 50.0, 0.99) if result.num_inliers > 0 else 0.1
                key = (round(cand["lat"], 6), round(cand["lng"], 6))
                img = retained_images.get(key)
                candidates.append(
                    LocationCandidate(
                        lat=cand["lat"],
                        lng=cand["lng"],
                        confidence=round(conf, 4),
                        agent=AgentName.STREETVIEW_MATCHER,
                        heading=cand["heading"],
                        evidence=(
                            f"LoFTR inliers={result.num_inliers} tentative={result.num_tentative} "
                            f"CLIP={cand['score']:.4f}"
                        ),
                        metadata={
                            "loftr_inliers": result.num_inliers,
                            "clip_score": round(cand["score"], 4),
                            "densified": True,
                            "_sv_image": img,
                        },
                    )
                )
        else:
            # ---- STAGE 4: Pure CLIP fallback (proven 95-137m) ----
            _log(f"[densify] All matchers inconclusive, using CLIP ranking")
            for m in scored[:10]:
                if m.image is not None:
                    retained_images[(round(m.lat, 6), round(m.lng, 6))] = m.image

            for m in scored[10:]:
                m.image = None
            for c in loftr_candidates:
                c.pop("image", None)

            from ..geo import cluster_scored_points
            clustered = cluster_scored_points(
                [(m.lat, m.lng, m.score, m.heading) for m in scored], merge_radius_m=25.0
            )
            sorted_scores = sorted(m.score for m in scored)

            candidates = []
            for pt in clustered[:8]:
                conf = percentile_confidence(pt.score, sorted_scores)
                key = (round(pt.lat, 6), round(pt.lng, 6))
                img = retained_images.get(key)
                candidates.append(
                    LocationCandidate(
                        lat=pt.lat,
                        lng=pt.lng,
                        confidence=conf,
                        agent=AgentName.STREETVIEW_MATCHER,
                        heading=pt.heading,
                        evidence=f"CLIP densify={pt.score:.4f} conf={conf:.2f}",
                        metadata={
                            "clip_score": round(pt.score, 4),
                            "densified": True,
                            "_sv_image": img,
                        },
                    )
                )

        method = "LoFTR" if loftr_discriminative else "CLIP"
        _log(
            f"[densify] done {_time.time() - started:.1f}s "
            f"method={method} best_inliers={best_inliers} clip_top={clip_top_score:.4f} candidates={len(candidates)}"
        )
        return AgentResult(
            agent=AgentName.STREETVIEW_MATCHER,
            candidates=candidates,
            notes=(
                f"Densification ({method}): {len(panos)} panos, {len(all_tasks)} frames, "
                f"best_inliers={best_inliers}, clip_top={clip_top_score:.4f}"
            ),
            runtime_s=_time.time() - started,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"[densify] error: {exc}")
        return AgentResult(
            agent=AgentName.STREETVIEW_MATCHER,
            candidates=[],
            error=str(exc),
            runtime_s=_time.time() - started,
        )
    finally:
        if own_client and client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
