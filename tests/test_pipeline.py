"""Regression tests for the contest-day pipeline changes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_cluster_heading_propagation():
    from doordash_geo_hunt.geo import cluster_scored_points

    pts = [
        (40.0, -74.0, 0.9, 123.0),
        (40.00001, -74.00001, 0.5, 45.0),  # ~1.5m away → merges with the above
        (41.0, -75.0, 0.7, 200.0),
    ]
    clusters = cluster_scored_points(pts, merge_radius_m=50.0)
    top = clusters[0]
    # Heading must come from the highest-scoring member, not float averaging.
    assert top.heading == 123.0
    assert top.members == 2


def test_percentile_confidence():
    from doordash_geo_hunt.agents.visual_matcher import percentile_confidence

    scores = sorted([0.1, 0.2, 0.3, 0.4, 0.5])
    low = percentile_confidence(0.1, scores)
    mid = percentile_confidence(0.3, scores)
    high = percentile_confidence(0.5, scores)
    assert 0.30 <= low <= mid <= high <= 0.95
    assert high > low  # ranking is monotonic


def test_ocr_blocklist():
    from doordash_geo_hunt.agents.vlm_agents import _filter_ocr_tokens

    tokens = ["THE", "BAG", "Starbucks", "go", "Main St", "DOORDASH", "Hi"]
    out = _filter_ocr_tokens(tokens)
    assert "Starbucks" in out
    assert "Main St" in out
    for banned in ("THE", "BAG", "DOORDASH", "go", "Hi"):
        assert banned not in out


def test_region_contains_filter():
    from doordash_geo_hunt.agents.vlm_agents import _extract_candidates
    from doordash_geo_hunt.models import AgentName, SearchRegion

    region = SearchRegion(center_lat=40.0, center_lng=-74.0, radius_m=500.0)
    text = (
        "```json\n"
        '[{"lat": 40.0, "lng": -74.0, "confidence": 0.9, "evidence": "in"},'
        ' {"lat": 41.0, "lng": -75.0, "confidence": 0.8, "evidence": "out"}]'
        "\n```"
    )
    cands = _extract_candidates(text, AgentName.VLM_GEOGUESSER, region)
    assert len(cands) == 1
    assert cands[0].evidence == "in"


def test_pano_dedup(monkeypatch):
    from doordash_geo_hunt.models import SearchRegion
    from doordash_geo_hunt.streetview import StreetViewClient

    client = StreetViewClient(api_key="dummy", workers=2)
    canned = {
        (0.0, 0.0): {"status": "OK", "pano_id": "A", "location": {"lat": 0.0, "lng": 0.0}},
        (0.001, 0.0): {"status": "OK", "pano_id": "A", "location": {"lat": 0.0, "lng": 0.0}},
        (0.002, 0.0): {"status": "OK", "pano_id": "B", "location": {"lat": 0.002, "lng": 0.0}},
    }
    monkeypatch.setattr(client, "has_panorama", lambda lat, lng: canned.get((lat, lng)))
    region = SearchRegion(center_lat=0.0, center_lng=0.0, radius_m=100_000.0)
    panos = client.grid_metadata(
        [(0.0, 0.0), (0.001, 0.0), (0.002, 0.0)], region=region, workers=2
    )
    client.close()
    assert sorted(p["pano_id"] for p in panos) == ["A", "B"]


def test_crop_location_background_shapes():
    from doordash_geo_hunt.preprocessing import crop_location_background

    rgba = np.zeros((100, 80, 4), dtype=np.uint8)
    assert crop_location_background(rgba).shape[2] == 3

    gray = np.zeros((100, 80), dtype=np.uint8)
    assert crop_location_background(gray).shape[2] == 3

    tiny = np.zeros((2, 2, 3), dtype=np.uint8)
    assert crop_location_background(tiny).shape[2] == 3

    # Synthetic image with a red bag in center-bottom → crop should exclude the bag
    img = np.full((200, 160, 3), 180, dtype=np.uint8)  # gray background
    img[120:190, 50:110] = [220, 30, 30]  # red bag region
    out = crop_location_background(img)
    assert out.ndim == 3 and out.shape[2] == 3
    # Output should be shorter than input (bag area cropped away)
    assert out.shape[0] < img.shape[0]

    sample = Path("samples/miami-drop1/photo3.jpg")
    if sample.exists():
        arr = np.asarray(Image.open(sample).convert("RGB"))
        out = crop_location_background(arr)
        assert out.ndim == 3 and out.shape[2] == 3
        assert out.shape[0] <= arr.shape[0]


def test_ingest_classify(tmp_path):
    from doordash_geo_hunt.twitter_fetcher import classify_photos

    # photo1 = dark promo
    Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8)).save(tmp_path / "photo1.jpg")
    # photo2 = map with a red zone (moderate redness ~0.2)
    map_img = np.full((500, 500, 3), 180, dtype=np.uint8)
    map_img[100:300, 100:300, 0] = 220  # red zone in center
    map_img[100:300, 100:300, 1] = 60
    map_img[100:300, 100:300, 2] = 60
    Image.fromarray(map_img).save(tmp_path / "photo2.jpg")
    # photo3 = location clue (neutral)
    Image.fromarray(np.full((600, 600, 3), 120, dtype=np.uint8)).save(tmp_path / "photo3.jpg")
    # photo4 = solid red promo (reddest overall — but classifier should NOT pick it)
    red = np.zeros((500, 500, 3), dtype=np.uint8)
    red[..., 0] = 220
    Image.fromarray(red).save(tmp_path / "photo4.jpg")

    res = classify_photos(tmp_path)
    # Canonical order wins: photo2=map, photo3=location (even though photo4 is redder)
    assert res["map"] == tmp_path / "photo2.jpg"
    assert res["location"] == tmp_path / "photo3.jpg"


def test_guess_city_patterns():
    from doordash_geo_hunt.twitter_fetcher import _guess_city

    assert _guess_city("New Jersey!! A DoorDash bag...") == "New Jersey"
    assert _guess_city("NEW YORK CITY!! tickets hidden") == "New York City"
    assert _guess_city("Find it in Dallas today") == "Dallas"
    assert _guess_city("Drop is live #Miami") == "Miami"
