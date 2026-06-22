from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from .geo import grid_points_in_circle, offset_lat_lng
from .models import SearchRegion


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def headings_evenly(n: int) -> list[float]:
    """N headings evenly spaced around 360°."""
    n = max(1, int(n))
    return [round(i * 360.0 / n, 1) for i in range(n)]


class StreetViewClient:
    METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
    STATIC_URL = "https://maps.googleapis.com/maps/api/streetview"

    def __init__(self, api_key: str | None = None, workers: int = 32) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY is required for Street View agent.")
        self.workers = max(1, int(workers))
        limits = httpx.Limits(
            max_connections=self.workers,
            max_keepalive_connections=self.workers,
        )
        self._client = httpx.Client(limits=limits, timeout=60, follow_redirects=True)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "StreetViewClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----- metadata (Phase 1) -------------------------------------------------
    def has_panorama(self, lat: float, lng: float) -> dict | None:
        params = {"location": f"{lat},{lng}", "key": self.api_key}
        resp = self._client.get(self.METADATA_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            return None
        return data

    def grid_metadata(
        self,
        points: list[tuple[float, float]],
        *,
        region: SearchRegion | None = None,
        max_panos: int = 600,
        workers: int | None = None,
    ) -> list[dict]:
        """Parallel metadata lookup over grid points, deduped by ``pano_id``."""
        workers = workers or self.workers
        seen: set[str] = set()
        lock = threading.Lock()
        panos: list[dict] = []

        def _probe(pt: tuple[float, float]) -> dict | None:
            try:
                meta = self.has_panorama(pt[0], pt[1])
            except Exception:  # noqa: BLE001
                return None
            if not meta:
                return None
            pano_id = meta.get("pano_id")
            loc = meta.get("location", {})
            lat = float(loc.get("lat", pt[0]))
            lng = float(loc.get("lng", pt[1]))
            if region is not None and not region.contains(lat, lng):
                return None
            return {"pano_id": pano_id, "lat": lat, "lng": lng}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_probe, points):
                if not result or not result["pano_id"]:
                    continue
                with lock:
                    if result["pano_id"] in seen:
                        continue
                    seen.add(result["pano_id"])
                    panos.append(result)
                    if len(panos) >= max_panos:
                        break
        return panos

    def list_panoramas(
        self,
        region: SearchRegion,
        step_m: float = 40.0,
        max_panos: int = 600,
        workers: int | None = None,
    ) -> list[dict]:
        points = grid_points_in_circle(
            region.center_lat, region.center_lng, region.radius_m, step_m
        )
        _log(f"[sv] grid_points={len(points)} step_m={step_m:.0f}")
        panos = self.grid_metadata(points, region=region, max_panos=max_panos, workers=workers)
        _log(f"[sv] unique_panos={len(panos)}")
        return panos

    def panoramas_near(
        self,
        centers: list[tuple[float, float]],
        *,
        region: SearchRegion,
        radius_m: float = 60.0,
        step_m: float = 25.0,
        max_panos: int = 200,
        workers: int | None = None,
    ) -> list[dict]:
        """Unique panoramas within ``radius_m`` of each center point."""
        points: list[tuple[float, float]] = []
        n = max(1, int(radius_m / step_m))
        for clat, clng in centers:
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    north, east = i * step_m, j * step_m
                    if (north**2 + east**2) ** 0.5 > radius_m:
                        continue
                    lat, lng = offset_lat_lng(clat, clng, north, east)
                    if region.contains(lat, lng):
                        points.append((lat, lng))
        return self.grid_metadata(points, region=region, max_panos=max_panos, workers=workers)

    # ----- image fetch (Phase 2) ---------------------------------------------
    def fetch_image(
        self,
        lat: float,
        lng: float,
        heading: float = 0,
        pitch: float = 0,
        fov: int = 90,
        width: int = 640,
        height: int = 640,
    ) -> Image.Image | None:
        params = {
            "location": f"{lat},{lng}",
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "size": f"{width}x{height}",
            "key": self.api_key,
        }
        resp = self._client.get(self.STATIC_URL, params=params)
        if resp.status_code != 200:
            return None
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def fetch_frames(
        self,
        tasks: list[dict],
        *,
        fov: int = 90,
        workers: int | None = None,
        cache_dir: Path | None = None,
        label: str = "sv",
        progress_every: int = 50,
    ) -> list[dict]:
        """Fetch many framed views in parallel.

        Each task: ``{lat, lng, heading, pitch, pano_id}``. Returns the same dicts
        with an added ``image`` (PIL). Disk is only written when ``cache_dir`` is
        provided (dev only — contest runs fetch fresh).
        """
        if not tasks:
            return []
        workers = workers or self.workers
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        total = len(tasks)
        done = 0
        lock = threading.Lock()
        out: list[dict] = []

        def _fetch(task: dict) -> dict | None:
            img = self.fetch_image(
                task["lat"],
                task["lng"],
                heading=task.get("heading", 0.0),
                pitch=task.get("pitch", 0.0),
                fov=fov,
            )
            if img is None:
                return None
            entry = dict(task)
            entry["image"] = img
            if cache_dir and task.get("pano_id"):
                fname = cache_dir / (
                    f"{task['pano_id']}_{int(task.get('heading', 0))}"
                    f"_{int(task.get('pitch', 0))}.jpg"
                )
                if not fname.exists():
                    img.save(fname, quality=85)
                entry["path"] = fname
            return entry

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_fetch, t) for t in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                with lock:
                    done += 1
                    if done % progress_every == 0 or done == total:
                        pct = int(100 * done / total)
                        _log(f"[{label}] {done}/{total} ({pct}%) workers={workers}")
                if res is not None:
                    out.append(res)
        return out
