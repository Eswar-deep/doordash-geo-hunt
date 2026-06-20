from __future__ import annotations

from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from .geo import haversine_m
from .models import SearchRegion

NEARBY_PHOTOS_URL = "https://api.openstreetcam.org/1.0/list/nearby-photos/"
PHOTO_DETAIL_URL = "https://api.openstreetcam.org/2.0/photo/"


class KartaViewClient:
    """Public KartaView / OpenStreetCam API — no API key required."""

    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout

    def images_in_region(
        self,
        region: SearchRegion,
        limit: int = 120,
        cache_dir: Path | None = None,
        max_attempts: int = 80,
    ) -> list[dict]:
        """
        Fetch street-level photos near the circle center, filtered to inside the circle.

        Uses the public POST /1.0/list/nearby-photos/ endpoint (no auth).
        Image CDN links can 404/502 for older sequences; those frames are skipped.
        """
        radius_m = min(int(region.radius_m), 1000)
        resp = httpx.post(
            NEARBY_PHOTOS_URL,
            data={
                "lat": region.center_lat,
                "lng": region.center_lng,
                "radius": radius_m,
            },
            timeout=self.timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("currentPageItems") or payload.get("result", {}).get("data") or []

        samples: list[dict] = []
        seen_ids: set[str] = set()
        attempts = 0
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            if len(samples) >= limit or attempts >= max_attempts:
                break
            attempts += 1

            photo_id = str(item.get("id", ""))
            if not photo_id or photo_id in seen_ids:
                continue

            lat = float(item.get("match_lat") or item.get("lat"))
            lng = float(item.get("match_lng") or item.get("lng"))
            if not region.contains(lat, lng):
                continue

            image = self._download_image(item)
            if image is None:
                continue

            seen_ids.add(photo_id)
            heading = item.get("heading") or item.get("headers")
            entry: dict = {
                "lat": lat,
                "lng": lng,
                "image_id": photo_id,
                "image": image,
                "heading": float(heading) if heading is not None else None,
            }
            if cache_dir:
                path = cache_dir / f"{photo_id}.jpg"
                if not path.exists():
                    image.save(path, quality=85)
                entry["path"] = path
            samples.append(entry)

        return samples

    def _download_image(self, item: dict) -> Image.Image | None:
        for url in self._image_urls(item):
            try:
                resp = httpx.get(url, timeout=8, follow_redirects=True)
            except httpx.HTTPError:
                continue
            if resp.status_code != 200 or len(resp.content) < 1000:
                continue
            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type and not url.endswith(".jpg"):
                continue
            return Image.open(BytesIO(resp.content)).convert("RGB")
        return None

    def _image_urls(self, item: dict) -> list[str]:
        urls: list[str] = []
        for name_key in ("th_name", "lth_name", "name"):
            name = item.get(name_key)
            if not name or not isinstance(name, str):
                continue
            urls.append(self._storage_url(name))

        # Fallback: one detail lookup only if list payload had no usable paths
        if not urls:
            detail = self._photo_detail(str(item.get("id", "")))
            if detail:
                for key in ("fileurlTh", "fileurlLTh", "fileurlProc"):
                    url = detail.get(key)
                    if url and "{{" not in url:
                        urls.append(url)

        seen: set[str] = set()
        ordered: list[str] = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                ordered.append(url)
        return ordered

    def _photo_detail(self, photo_id: str) -> dict | None:
        try:
            resp = httpx.get(
                PHOTO_DETAIL_URL,
                params={"id": photo_id},
                timeout=20,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return None
            data = resp.json().get("result", {}).get("data") or []
            return data[0] if data else None
        except httpx.HTTPError:
            return None

    @staticmethod
    def _storage_url(relative_path: str) -> str:
        # e.g. storage7/files/photo/2018/3/24/th/foo.jpg
        parts = relative_path.split("/", 1)
        if len(parts) == 2 and parts[0].startswith("storage"):
            return f"https://{parts[0]}.openstreetcam.org/{parts[1]}"
        return f"https://storage.openstreetcam.org/files/photo/{relative_path}"
