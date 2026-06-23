"""Local feature matching using LoFTR for definitive location verification.

Unlike CLIP (global embedding similarity), LoFTR finds SPECIFIC keypoint
correspondences between images. Two views of the same physical location will
share many geometrically consistent keypoints (pipe bends, brick corners,
mortar lines). Two visually similar but different locations will share near zero.

Pipeline: CLIP narrows 38K frames → top 200. LoFTR re-ranks those 200 by
counting geometrically verified inlier matches. The frame with most inliers
is definitively the correct location.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class FeatureMatchResult:
    num_inliers: int
    num_tentative: int
    score: float  # inliers / max_possible, clamped to [0, 1]


_LOCK = threading.Lock()
_MATCHER = None


def _get_matcher():
    """Lazy-load LoFTR model (shared singleton)."""
    global _MATCHER
    if _MATCHER is not None:
        return _MATCHER
    with _LOCK:
        if _MATCHER is not None:
            return _MATCHER
        import kornia.feature as KF
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _log(f"[loftr] Loading LoFTR (outdoor) on {device}...")
        model = KF.LoFTR(pretrained="outdoor")
        model = model.eval().to(device)
        _MATCHER = model
        _log("[loftr] Ready")
        return _MATCHER


def _pil_to_tensor(img: Image.Image, max_side: int = 480) -> torch.Tensor:
    """Convert PIL image to grayscale tensor for LoFTR, resized to fit max_side."""
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = img_rgb.resize((new_w, new_h), Image.LANCZOS)

    arr = np.array(img_rgb).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    import kornia as K
    gray = K.color.rgb_to_grayscale(tensor)  # [1, 1, H, W]
    return gray


def match_pair(query: Image.Image, candidate: Image.Image) -> FeatureMatchResult:
    """Count geometrically verified keypoint matches between query and candidate.

    Returns FeatureMatchResult with inlier count (higher = better match).
    A true match typically has 30-200+ inliers; wrong locations have 0-5.
    """
    matcher = _get_matcher()
    device = next(matcher.parameters()).device

    img0 = _pil_to_tensor(query, max_side=480).to(device)
    img1 = _pil_to_tensor(candidate, max_side=480).to(device)

    with torch.inference_mode():
        correspondences = matcher({"image0": img0, "image1": img1})

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    confidence = correspondences["confidence"].cpu().numpy()

    num_tentative = len(mkpts0)
    if num_tentative < 4:
        return FeatureMatchResult(num_inliers=0, num_tentative=num_tentative, score=0.0)

    # RANSAC geometric verification — only count spatially consistent matches
    _, inliers_mask = cv2.findFundamentalMat(
        mkpts0, mkpts1, cv2.USAC_MAGSAC, ransacReprojThreshold=1.0, confidence=0.999, maxIters=10000
    )
    num_inliers = int(inliers_mask.sum()) if inliers_mask is not None else 0

    score = min(num_inliers / 50.0, 1.0)  # 50+ inliers = perfect score
    return FeatureMatchResult(num_inliers=num_inliers, num_tentative=num_tentative, score=score)


def rerank_candidates(
    query: Image.Image,
    candidates: list[dict],
    top_k: int = 10,
) -> list[tuple[int, FeatureMatchResult]]:
    """Re-rank candidate frames by LoFTR inlier count.

    Args:
        query: The clue/query image (background-cropped).
        candidates: List of dicts with at minimum an "image" key (PIL Image).
        top_k: Return top-K results.

    Returns:
        List of (original_index, FeatureMatchResult) sorted by inlier count descending.
    """
    results: list[tuple[int, FeatureMatchResult]] = []

    for i, cand in enumerate(candidates):
        img = cand.get("image")
        if img is None:
            results.append((i, FeatureMatchResult(num_inliers=0, num_tentative=0, score=0.0)))
            continue

        result = match_pair(query, img)
        results.append((i, result))

        if (i + 1) % 20 == 0:
            _log(f"[loftr] {i + 1}/{len(candidates)} scored, best so far: {max(r.num_inliers for _, r in results)} inliers")

    results.sort(key=lambda x: x[1].num_inliers, reverse=True)

    if results:
        _log(
            f"[loftr] done: top={results[0][1].num_inliers} inliers, "
            f"2nd={results[1][1].num_inliers if len(results) > 1 else 0}, "
            f"3rd={results[2][1].num_inliers if len(results) > 2 else 0}"
        )

    return results[:top_k]
