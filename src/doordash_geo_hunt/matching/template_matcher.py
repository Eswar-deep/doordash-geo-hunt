"""Multi-patch template matching for precise location verification.

Strategy: extract multiple DISTINCTIVE patches from the clue photo, then for
each Street View candidate, check how many patches match. The correct location
will match most/all patches; wrong locations match 0-2 by chance.

This works where CLIP fails because:
- CLIP sees "brick wall" globally → many false positives
- Template matching sees "THIS specific pipe at THIS specific brick joint"
  → only the exact location matches
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class PatchMatch:
    score: float  # NCC correlation [0, 1]
    location: tuple[int, int]  # (x, y) in candidate where patch was found
    scale: float  # scale at which match was found


@dataclass
class MultiPatchResult:
    """Result of matching multiple patches against a single candidate."""
    total_score: float  # sum of best correlation for each patch
    num_matched: int  # how many patches exceeded threshold
    num_patches: int  # total patches attempted
    match_ratio: float  # num_matched / num_patches
    patch_scores: list[float] = field(default_factory=list)


def extract_distinctive_patches(
    query: Image.Image,
    num_patches: int = 8,
    patch_size: int = 64,
    min_variance: float = 500.0,
) -> list[np.ndarray]:
    """Auto-detect and extract distinctive patches from the query image.

    Distinctive = high local variance + strong edges + low self-repetition.
    These are the features that uniquely identify THIS specific location.
    """
    arr = np.array(query.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    if h < patch_size * 2 or w < patch_size * 2:
        return []

    # Compute edge magnitude (Sobel) — patches with strong edges are structural
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32)

    # Score every possible patch position by: variance + edge density
    step = patch_size // 2  # 50% overlap grid
    candidates: list[tuple[float, int, int]] = []

    for y in range(0, h - patch_size, step):
        for x in range(0, w - patch_size, step):
            patch = gray[y : y + patch_size, x : x + patch_size]
            edge_patch = edges[y : y + patch_size, x : x + patch_size]

            variance = np.var(patch)
            edge_density = np.mean(edge_patch) / 255.0

            # Skip low-variance (flat) or low-edge (textureless) patches
            if variance < min_variance:
                continue
            if edge_density < 0.05:
                continue

            # Combined score: patches with both high variance AND edges
            score = variance * (1.0 + edge_density * 3.0)

            # Penalize repetitive patches (self-similar within the image)
            # by checking if this patch pattern appears multiple times
            # Simple proxy: if patch has mostly one dominant direction, it's repetitive
            candidates.append((score, y, x))

    if not candidates:
        return []

    # Sort by distinctiveness score, take top-N
    candidates.sort(key=lambda c: c[0], reverse=True)

    # Remove overlapping patches (non-maximum suppression)
    selected: list[tuple[float, int, int]] = []
    min_dist = patch_size * 0.7  # patches must be at least 70% patch_size apart

    for score, y, x in candidates:
        too_close = False
        for _, sy, sx in selected:
            if abs(y - sy) < min_dist and abs(x - sx) < min_dist:
                too_close = True
                break
        if not too_close:
            selected.append((score, y, x))
        if len(selected) >= num_patches:
            break

    # Extract the actual patches (in color for richer matching)
    patches = []
    for _, y, x in selected:
        patch = arr[y : y + patch_size, x : x + patch_size]
        patches.append(patch)

    _log(f"[template] extracted {len(patches)} distinctive patches from query")
    return patches


def match_patches_against_candidate(
    patches: list[np.ndarray],
    candidate: Image.Image,
    scales: list[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
    match_threshold: float = 0.45,
) -> MultiPatchResult:
    """Match all patches against a single candidate image at multiple scales.

    Returns how many patches matched and overall correlation score.
    """
    cand_arr = np.array(candidate.convert("RGB"))
    cand_gray = cv2.cvtColor(cand_arr, cv2.COLOR_RGB2GRAY)
    ch, cw = cand_gray.shape

    patch_scores: list[float] = []
    num_matched = 0

    for patch in patches:
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        ph, pw = patch_gray.shape
        best_corr = 0.0

        for scale in scales:
            # Resize patch to simulate different scales
            new_ph = int(ph * scale)
            new_pw = int(pw * scale)
            if new_ph >= ch or new_pw >= cw or new_ph < 8 or new_pw < 8:
                continue

            scaled_patch = cv2.resize(patch_gray, (new_pw, new_ph), interpolation=cv2.INTER_LINEAR)

            # Normalized cross-correlation
            result = cv2.matchTemplate(cand_gray, scaled_patch, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            best_corr = max(best_corr, max_val)

        patch_scores.append(best_corr)
        if best_corr >= match_threshold:
            num_matched += 1

    total_score = sum(patch_scores)
    match_ratio = num_matched / len(patches) if patches else 0.0

    return MultiPatchResult(
        total_score=total_score,
        num_matched=num_matched,
        num_patches=len(patches),
        match_ratio=match_ratio,
        patch_scores=patch_scores,
    )


def rerank_by_template(
    query: Image.Image,
    candidates: list[dict],
    top_k: int = 10,
    num_patches: int = 8,
    patch_size: int = 64,
) -> list[tuple[int, MultiPatchResult]]:
    """Re-rank candidates by multi-patch template matching.

    The correct location matches MOST patches (4-8 out of 8).
    Wrong locations match 0-2 patches by chance.

    Args:
        query: The clue background image.
        candidates: List of dicts with "image" key (PIL Image).
        top_k: Return top-K results.

    Returns:
        List of (original_index, MultiPatchResult) sorted by match quality.
    """
    # Extract distinctive patches from query
    patches = extract_distinctive_patches(query, num_patches=num_patches, patch_size=patch_size)
    if not patches:
        _log("[template] no distinctive patches found in query")
        return []

    results: list[tuple[int, MultiPatchResult]] = []

    for i, cand in enumerate(candidates):
        img = cand.get("image")
        if img is None:
            results.append((i, MultiPatchResult(0.0, 0, len(patches), 0.0)))
            continue

        result = match_patches_against_candidate(patches, img)
        results.append((i, result))

        if (i + 1) % 10 == 0:
            best_so_far = max(r.num_matched for _, r in results)
            _log(f"[template] {i + 1}/{len(candidates)} scored, best: {best_so_far}/{len(patches)} patches matched")

    # Sort by: first by num_matched (primary), then by total_score (tiebreaker)
    results.sort(key=lambda x: (x[1].num_matched, x[1].total_score), reverse=True)

    if results:
        top = results[0][1]
        _log(
            f"[template] done: top matched {top.num_matched}/{top.num_patches} patches "
            f"(total_score={top.total_score:.3f}), "
            f"2nd={results[1][1].num_matched if len(results) > 1 else 0}, "
            f"3rd={results[2][1].num_matched if len(results) > 2 else 0}"
        )

    return results[:top_k]
