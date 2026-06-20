from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from ..geo import cluster_candidates
from ..kartaview import KartaViewClient
from ..mapillary import MapillaryClient
from ..matching.clip_matcher import ClipMatcher
from ..models import AgentName, AgentResult, ContestInput, LocationCandidate, SearchRegion
from ..preprocessing import crop_location_background, enhance_for_matching, load_rgb, resize_max_side


def _query_image(contest: ContestInput) -> Image.Image:
    raw = load_rgb(contest.location_image)
    cropped = crop_location_background(raw)
    enhanced = enhance_for_matching(cropped)
    return Image.fromarray(resize_max_side(enhanced))


def run_streetview_matcher(
    contest: ContestInput,
    region: SearchRegion,
    cache_dir: Path | None = None,
    step_m: float = 40.0,
) -> AgentResult:
    started = time.time()
    try:
        from ..streetview import StreetViewClient

        client = StreetViewClient()
        matcher = ClipMatcher()
        query = _query_image(contest)
        sv_cache = (cache_dir / "streetview") if cache_dir else None
        samples = client.sample_panoramas(region, step_m=step_m, cache_dir=sv_cache)
        ranked = matcher.rank(query, samples, top_k=20)
        triples = [(m.lat, m.lng, m.score) for m in ranked]
        merged = cluster_candidates(triples, merge_radius_m=30.0)

        candidates = [
            LocationCandidate(
                lat=lat,
                lng=lng,
                confidence=min(0.99, max(0.0, (score + 1) / 2)),
                agent=AgentName.STREETVIEW_MATCHER,
                heading=next((m.heading for m in ranked if m.lat == lat and m.lng == lng), None),
                evidence=f"CLIP street-view match score={score:.4f}",
            )
            for lat, lng, score in merged[:8]
        ]
        return AgentResult(
            agent=AgentName.STREETVIEW_MATCHER,
            candidates=candidates,
            notes=f"Sampled {len(samples)} street-view frames in circle.",
            runtime_s=time.time() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            agent=AgentName.STREETVIEW_MATCHER,
            candidates=[],
            error=str(exc),
            runtime_s=time.time() - started,
        )


def run_kartaview_matcher(
    contest: ContestInput,
    region: SearchRegion,
    cache_dir: Path | None = None,
) -> AgentResult:
    started = time.time()
    try:
        client = KartaViewClient()
        matcher = ClipMatcher()
        query = _query_image(contest)
        kv_cache = (cache_dir / "kartaview") if cache_dir else None
        samples = client.images_in_region(region, cache_dir=kv_cache)
        ranked = matcher.rank(query, samples, top_k=20)
        merged = cluster_candidates([(m.lat, m.lng, m.score) for m in ranked], merge_radius_m=30.0)

        candidates = [
            LocationCandidate(
                lat=lat,
                lng=lng,
                confidence=min(0.99, max(0.0, (score + 1) / 2)),
                agent=AgentName.KARTAVIEW_MATCHER,
                heading=next((m.heading for m in ranked if m.lat == lat and m.lng == lng), None),
                evidence=f"CLIP KartaView match score={score:.4f}",
            )
            for lat, lng, score in merged[:8]
        ]
        return AgentResult(
            agent=AgentName.KARTAVIEW_MATCHER,
            candidates=candidates,
            notes=f"Compared against {len(samples)} KartaView frames (public API, no key).",
            runtime_s=time.time() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            agent=AgentName.KARTAVIEW_MATCHER,
            candidates=[],
            error=str(exc),
            runtime_s=time.time() - started,
        )


def run_mapillary_matcher(
    contest: ContestInput,
    region: SearchRegion,
    cache_dir: Path | None = None,
) -> AgentResult:
    started = time.time()
    try:
        client = MapillaryClient()
        matcher = ClipMatcher()
        query = _query_image(contest)
        m_cache = (cache_dir / "mapillary") if cache_dir else None
        samples = client.images_in_region(region, cache_dir=m_cache)
        ranked = matcher.rank(query, samples, top_k=20)
        merged = cluster_candidates([(m.lat, m.lng, m.score) for m in ranked], merge_radius_m=30.0)

        candidates = [
            LocationCandidate(
                lat=lat,
                lng=lng,
                confidence=min(0.99, max(0.0, (score + 1) / 2)),
                agent=AgentName.MAPILLARY_MATCHER,
                evidence=f"CLIP Mapillary match score={score:.4f}",
            )
            for lat, lng, score in merged[:8]
        ]
        return AgentResult(
            agent=AgentName.MAPILLARY_MATCHER,
            candidates=candidates,
            notes=f"Compared against {len(samples)} Mapillary frames.",
            runtime_s=time.time() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            agent=AgentName.MAPILLARY_MATCHER,
            candidates=[],
            error=str(exc),
            runtime_s=time.time() - started,
        )
