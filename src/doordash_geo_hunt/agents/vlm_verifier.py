"""VLM verification: compare top Street View candidates against the clue photo.

After densification produces CLIP's top candidates, this module fetches a
Street View frame for each and asks the VLM to visually compare them against
the original clue photo, picking the best match with reasoning.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

from ..models import AgentName, AgentResult, LocationCandidate
from ..pipeline_context import PipelineContext, StreetViewConfig


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.index("\n") if "\n" in text else 3
        text = text[first_nl + 1:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


_VERIFY_PROMPT = """You are comparing Street View images against a clue photo to find the exact location match.

Image 1 is the CLUE photo (a DoorDash bag on a pedestal — focus only on the BACKGROUND: walls, buildings, ground, pipes, fixtures, architectural details).

Images 2 through {n} are Street View captures from different nearby locations and angles.

For each Street View image (2-{n}), rate how well its background matches the clue on a scale of 0-100:
- 90-100: Near-perfect match (same wall, same pipe, same ground texture, clearly the spot)
- 70-89: Strong match (same building/wall type, similar features, likely the same location)
- 40-69: Partial match (similar architecture but different specific location)
- 0-39: Poor match (different building/scene entirely)

Pay attention to:
- Brick color, pattern, and weathering
- Drain pipes, gutters, fixtures on walls
- Ground surface (cobblestone, pavers, asphalt)
- Window/door styles and positions
- Wall-mounted lights or signs

Respond ONLY with this JSON (no other text):
{{
  "scores": [<score for image 2>, <score for image 3>, ...],
  "best_index": <0-based index of the best match in the scores array>,
  "confidence": <0-100 overall confidence that any of these is the correct spot>,
  "reasoning": "<one sentence explaining why the best match is correct or why none match>"
}}"""


def verify_candidates_with_vlm(
    ctx: PipelineContext,
    candidates: list[LocationCandidate],
    cfg: StreetViewConfig | None = None,
    max_verify: int = 5,
) -> AgentResult:
    """Fetch SV frames for top candidates and ask VLM which best matches the clue.

    Returns an AgentResult with the VLM-verified candidates re-ranked by
    the VLM's visual match score.
    """
    started = time.time()
    cfg = cfg or StreetViewConfig()

    if not candidates:
        return AgentResult(
            agent=AgentName.STREETVIEW_MATCHER,
            candidates=[],
            notes="No candidates to verify.",
            runtime_s=time.time() - started,
        )

    # Take top-N candidates for verification
    to_verify = candidates[:max_verify]
    _log(f"[vlm-verify] verifying {len(to_verify)} candidates against clue photo")

    own_client = False
    client = ctx.sv_client
    try:
        if client is None:
            from ..streetview import StreetViewClient
            client = StreetViewClient(workers=cfg.workers)
            own_client = True

        # Fetch one SV frame per candidate at the candidate's heading + pitch
        sv_images: list[tuple[Path, LocationCandidate]] = []
        tmp_dir = Path(tempfile.mkdtemp(prefix="vlm_verify_"))

        for i, cand in enumerate(to_verify):
            heading = cand.heading if cand.heading is not None else 0.0
            pitch = float(cand.metadata.get("pitch", 25.0)) if cand.metadata else 25.0
            img = client.fetch_image(
                cand.lat, cand.lng,
                heading=heading,
                pitch=pitch,
                fov=90,
                width=640,
                height=640,
            )
            if img is None:
                # Try pitch=0 as fallback
                img = client.fetch_image(
                    cand.lat, cand.lng,
                    heading=heading,
                    pitch=0.0,
                    fov=90,
                )
            if img is not None:
                path = tmp_dir / f"sv_candidate_{i}.jpg"
                img.save(path, quality=90)
                sv_images.append((path, cand))

        if not sv_images:
            _log("[vlm-verify] no SV images fetched successfully")
            return AgentResult(
                agent=AgentName.STREETVIEW_MATCHER,
                candidates=candidates,
                notes="VLM verify: no SV images fetched.",
                runtime_s=time.time() - started,
            )

        # Save clue photo as temp file
        clue_path = tmp_dir / "clue.jpg"
        ctx.query_image.save(clue_path, quality=95)

        # Build image list: [clue, sv1, sv2, ...]
        image_paths = [clue_path] + [p for p, _ in sv_images]
        n_images = len(image_paths)

        # Call VLM with multi-image prompt
        from ..llm_vision import vision_prompt_multi, VisionTask
        prompt = _VERIFY_PROMPT.format(n=n_images)
        _log(f"[vlm-verify] sending {n_images} images to VLM ({len(sv_images)} candidates)")
        response = vision_prompt_multi(prompt, image_paths, task=VisionTask.JUDGE)
        _log(f"[vlm-verify] VLM response: {response[:200]}")

        # Parse VLM response
        try:
            parsed = json.loads(_strip_code_fences(response))
            scores = parsed.get("scores", [])
            best_idx = parsed.get("best_index", 0)
            vlm_confidence = parsed.get("confidence", 50)
            reasoning = parsed.get("reasoning", "")
        except (json.JSONDecodeError, KeyError) as e:
            _log(f"[vlm-verify] parse error: {e}, response: {response[:300]}")
            return AgentResult(
                agent=AgentName.STREETVIEW_MATCHER,
                candidates=candidates,
                notes=f"VLM verify parse failed: {e}",
                runtime_s=time.time() - started,
            )

        # Re-rank candidates by VLM score
        verified_candidates: list[LocationCandidate] = []
        for i, (score_val, (_, cand)) in enumerate(zip(scores, sv_images)):
            vlm_score = float(score_val) / 100.0
            # Blend CLIP confidence with VLM score (VLM-dominant)
            blended_conf = round(0.3 * cand.confidence + 0.7 * vlm_score, 4)
            verified_candidates.append(
                LocationCandidate(
                    lat=cand.lat,
                    lng=cand.lng,
                    confidence=blended_conf,
                    agent=AgentName.STREETVIEW_MATCHER,
                    heading=cand.heading,
                    evidence=f"VLM-verified: score={score_val}/100 blended={blended_conf:.2f} | {cand.evidence}",
                    metadata={
                        **(cand.metadata or {}),
                        "vlm_verify_score": score_val,
                        "vlm_verified": True,
                    },
                )
            )

        # Sort by blended confidence (VLM-weighted)
        verified_candidates.sort(key=lambda c: c.confidence, reverse=True)

        _log(
            f"[vlm-verify] done {time.time() - started:.1f}s "
            f"best_idx={best_idx} vlm_conf={vlm_confidence} "
            f"top_score={scores[best_idx] if scores else 'N/A'}/100 "
            f"reasoning: {reasoning}"
        )

        return AgentResult(
            agent=AgentName.STREETVIEW_MATCHER,
            candidates=verified_candidates,
            notes=(
                f"VLM-verified {len(sv_images)} candidates. "
                f"Best: idx={best_idx} score={scores[best_idx] if scores else 'N/A'}/100 "
                f"conf={vlm_confidence}%. {reasoning}"
            ),
            runtime_s=time.time() - started,
        )

    except Exception as exc:  # noqa: BLE001
        _log(f"[vlm-verify] error: {exc}")
        return AgentResult(
            agent=AgentName.STREETVIEW_MATCHER,
            candidates=candidates,
            notes=f"VLM verify failed: {exc}",
            error=str(exc),
            runtime_s=time.time() - started,
        )
    finally:
        if own_client and client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
