from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from .geo import grid_points_in_circle
from .models import SearchRegion


class StreetViewClient:
    METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
    STATIC_URL = "https://maps.googleapis.com/maps/api/streetview"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY is required for Street View agent.")

    def has_panorama(self, lat: float, lng: float) -> dict | None:
        params = {"location": f"{lat},{lng}", "key": self.api_key}
        resp = httpx.get(self.METADATA_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            return None
        return data

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
        resp = httpx.get(self.STATIC_URL, params=params, timeout=60)
        if resp.status_code != 200:
            return None
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def sample_panoramas(
        self,
        region: SearchRegion,
        step_m: float = 40.0,
        headings: tuple[float, ...] = (0, 90, 180, 270),
        cache_dir: Path | None = None,
    ) -> list[dict]:
        points = grid_points_in_circle(region.center_lat, region.center_lng, region.radius_m, step_m)
        samples: list[dict] = []
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        for lat, lng in points:
            meta = self.has_panorama(lat, lng)
            if not meta:
                continue
            pano_lat = meta.get("location", {}).get("lat", lat)
            pano_lng = meta.get("location", {}).get("lng", lng)
            pano_id = meta.get("pano_id")
            for heading in headings:
                image = self.fetch_image(pano_lat, pano_lng, heading=heading)
                if image is None:
                    continue
                entry = {
                    "lat": float(pano_lat),
                    "lng": float(pano_lng),
                    "heading": heading,
                    "pano_id": pano_id,
                    "image": image,
                }
                if cache_dir and pano_id:
                    fname = cache_dir / f"{pano_id}_{int(heading)}.jpg"
                    if not fname.exists():
                        image.save(fname, quality=85)
                    entry["path"] = fname
                samples.append(entry)
        return samples
