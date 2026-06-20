from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from .geo import haversine_m
from .models import SearchRegion


class MapillaryClient:
    GRAPH_URL = "https://graph.mapillary.com"

    def __init__(self, access_token: str | None = None) -> None:
        self.access_token = access_token or os.getenv("MAPILLARY_ACCESS_TOKEN")
        if not self.access_token:
            raise RuntimeError("MAPILLARY_ACCESS_TOKEN is required for Mapillary agent.")

    def images_in_region(
        self,
        region: SearchRegion,
        limit: int = 120,
        cache_dir: Path | None = None,
    ) -> list[dict]:
        # Approximate bbox from circle
        lat_delta = region.radius_m / 111_000
        lng_delta = region.radius_m / (111_000 * max(0.2, abs(__import__("math").cos(__import__("math").radians(region.center_lat)))))
        bbox = {
            "bbox": f"{region.center_lng - lng_delta},{region.center_lat - lat_delta},"
            f"{region.center_lng + lng_delta},{region.center_lat + lat_delta}",
            "limit": limit,
            "fields": "id,computed_geometry,thumb_256_url,captured_at",
            "access_token": self.access_token,
        }
        resp = httpx.get(f"{self.GRAPH_URL}/images", params=bbox, timeout=60)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        samples: list[dict] = []
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        for item in data:
            geom = item.get("computed_geometry", {})
            coords = geom.get("coordinates")
            if not coords or len(coords) < 2:
                continue
            lng, lat = coords[0], coords[1]
            if haversine_m(region.center_lat, region.center_lng, lat, lng) > region.radius_m:
                continue
            thumb = item.get("thumb_256_url")
            if not thumb:
                continue
            img_resp = httpx.get(thumb, timeout=30)
            if img_resp.status_code != 200:
                continue
            image = Image.open(BytesIO(img_resp.content)).convert("RGB")
            image_id = item["id"]
            entry = {"lat": lat, "lng": lng, "image_id": image_id, "image": image}
            if cache_dir:
                path = cache_dir / f"{image_id}.jpg"
                if not path.exists():
                    image.save(path, quality=85)
                entry["path"] = path
            samples.append(entry)
        return samples
