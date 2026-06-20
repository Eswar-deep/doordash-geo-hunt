from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AgentName(str, Enum):
    STREETVIEW_MATCHER = "streetview_matcher"
    MAPILLARY_MATCHER = "mapillary_matcher"
    KARTAVIEW_MATCHER = "kartaview_matcher"
    VLM_GEOGUESSER = "vlm_geoguesser"
    LANDMARK_OCR = "landmark_ocr"


@dataclass(frozen=True)
class SearchRegion:
    """Circular search area extracted from the map screenshot."""

    center_lat: float
    center_lng: float
    radius_m: float
    city: str | None = None
    source: str = "manual"

    def contains(self, lat: float, lng: float) -> bool:
        from .geo import haversine_m

        return haversine_m(self.center_lat, self.center_lng, lat, lng) <= self.radius_m


@dataclass
class LocationCandidate:
    lat: float
    lng: float
    confidence: float
    agent: AgentName
    heading: float | None = None
    pitch: float | None = None
    evidence: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": self.lat,
            "lng": self.lng,
            "confidence": self.confidence,
            "agent": self.agent.value,
            "heading": self.heading,
            "pitch": self.pitch,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass
class AgentResult:
    agent: AgentName
    candidates: list[LocationCandidate]
    notes: str = ""
    runtime_s: float = 0.0
    error: str | None = None

    @property
    def best(self) -> LocationCandidate | None:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.confidence)


@dataclass
class ContestInput:
    map_image: Path
    location_image: Path
    city_hint: str | None = None
    region_override: SearchRegion | None = None

    def validate(self) -> None:
        if not self.map_image.exists():
            raise FileNotFoundError(f"Map image not found: {self.map_image}")
        if not self.location_image.exists():
            raise FileNotFoundError(f"Location image not found: {self.location_image}")


@dataclass
class FinalVerdict:
    lat: float
    lng: float
    confidence: float
    reasoning: str
    winner_agent: AgentName | None
    all_candidates: list[LocationCandidate]
    street_view_url: str | None = None
