from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def offset_lat_lng(lat: float, lng: float, north_m: float, east_m: float) -> tuple[float, float]:
    dlat = north_m / EARTH_RADIUS_M
    dlng = east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lng + math.degrees(dlng)


def grid_points_in_circle(
    center_lat: float,
    center_lng: float,
    radius_m: float,
    step_m: float = 35.0,
) -> list[tuple[float, float]]:
    """Generate a lat/lng grid covering the circle (used for Street View sampling)."""
    points: list[tuple[float, float]] = []
    n_steps = int(math.ceil(radius_m / step_m))
    for i in range(-n_steps, n_steps + 1):
        for j in range(-n_steps, n_steps + 1):
            north = i * step_m
            east = j * step_m
            if math.hypot(north, east) > radius_m:
                continue
            lat, lng = offset_lat_lng(center_lat, center_lng, north, east)
            points.append((lat, lng))
    return points


def cluster_candidates(
    candidates: list[tuple[float, float, float]],
    merge_radius_m: float = 25.0,
) -> list[tuple[float, float, float]]:
    """Merge nearby (lat, lng, score) tuples by weighted average."""
    if not candidates:
        return []

    clusters: list[list[tuple[float, float, float]]] = []
    for lat, lng, score in sorted(candidates, key=lambda x: x[2], reverse=True):
        placed = False
        for cluster in clusters:
            clat, clng, _ = cluster[0]
            if haversine_m(lat, lng, clat, clng) <= merge_radius_m:
                cluster.append((lat, lng, score))
                placed = True
                break
        if not placed:
            clusters.append([(lat, lng, score)])

    merged: list[tuple[float, float, float]] = []
    for cluster in clusters:
        total = sum(s for _, _, s in cluster)
        if total <= 0:
            continue
        lat = sum(lat * s for lat, _, s in cluster) / total
        lng = sum(lng * s for _, lng, s in cluster) / total
        score = max(s for _, _, s in cluster)
        merged.append((lat, lng, score))
    return sorted(merged, key=lambda x: x[2], reverse=True)
