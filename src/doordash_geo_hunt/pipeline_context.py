from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .models import ContestInput, SearchRegion


@dataclass
class StreetViewConfig:
    """Tuning for the Street View matcher's coarse→fine→refine sweep."""

    coarse_fine: bool = True
    headings_coarse: int = 8
    headings_fine: int = 12
    headings_override: int | None = None  # used when coarse_fine is off
    heading_step: int | None = None  # derive heading count from a degree step
    refine_headings: bool = True
    refine_span: int = 40
    refine_step: int = 5
    pitch_refine: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 40.0, -10.0)
    max_frames: int = 10000
    step_m: float | None = None
    workers: int = 48
    clip_batch_size: int = 128
    cache: bool = False
    fov_coarse: int = 120
    fov_fine: int = 90
    refine_max_frames: int = 150
    max_panos: int = 1200
    headings: int = 24
    # Pitch values for the exhaustive pass — 3 tilt levels cover
    # ground-level (0°), mid-tilt (20°), and steep-tilt (40°) views
    pitch_sweep: tuple[float, ...] = (0.0, 20.0, 40.0)


@dataclass
class PipelineContext:
    """Shared, preprocessed state built once on the main thread before agents fan out.

    Building this once avoids re-preprocessing the location photo per agent and
    avoids concurrent first-import of torch/open_clip across worker threads.
    """

    contest: ContestInput
    region: SearchRegion
    query_image: Image.Image
    query_crops: list | None = None  # list[Image.Image] auxiliary crops for multi-crop embedding
    clip_matcher: object | None = None  # ClipMatcher (avoid heavy import at module load)
    sv_client: object | None = None  # StreetViewClient | None
