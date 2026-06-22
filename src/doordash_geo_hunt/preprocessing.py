from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Coerce any array into an HxWx3 uint8 RGB image."""
    if image.ndim == 2:  # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _detect_bag_mask(image: np.ndarray) -> np.ndarray:
    """Detect the DoorDash bag/pedestal foreground using color + position heuristics.

    The DoorDash bag is typically:
    - Red/orange in color (high R, low G/B)
    - Located in the center-bottom of the frame
    - The largest warm-colored connected component in that region

    Returns a binary mask (uint8, 0 or 255) of the bag region to be masked out.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Detect red/orange hues (DoorDash red wraps around H=0 in HSV).
    # Low-range red: H 0-15, high-range red: H 160-180
    mask_lo = cv2.inRange(hsv, (0, 60, 50), (18, 255, 255))
    mask_hi = cv2.inRange(hsv, (155, 60, 50), (180, 255, 255))
    red_mask = mask_lo | mask_hi

    # Also catch darker reds/maroons that appear in shadows
    mask_dark = cv2.inRange(hsv, (0, 40, 25), (20, 255, 120))
    red_mask = red_mask | mask_dark

    # Weight toward center-bottom: the bag is rarely in the top 25% or outer 15% edges
    position_weight = np.zeros((h, w), dtype=np.uint8)
    y_start = int(h * 0.20)
    x_margin = int(w * 0.12)
    position_weight[y_start:, x_margin : w - x_margin] = 255
    red_mask = red_mask & position_weight

    # Morphological cleanup: close small gaps, then dilate to cover pedestal/shadow
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)

    # Find the largest connected component (the bag)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros((h, w), dtype=np.uint8)

    # Skip label 0 (background), find largest by area
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    bag_area = areas[largest_label - 1]

    # Minimum area threshold: bag should be at least 2% of image area
    if bag_area < h * w * 0.02:
        return np.zeros((h, w), dtype=np.uint8)

    bag_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Dilate generously to cover the pedestal, shadow, and any non-red parts of the bag
    dilate_px = max(int(min(h, w) * 0.06), 10)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    bag_mask = cv2.dilate(bag_mask, kernel_dilate)

    # Extend the mask downward to the bottom of the image (pedestal + floor below bag)
    contours, _ = cv2.findContours(bag_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x_min = min(cv2.boundingRect(c)[0] for c in contours)
        x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
        y_max_contour = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)
        # Fill from the bottom of detected bag to image bottom
        if y_max_contour < h:
            bag_mask[y_max_contour:, x_min:x_max] = 255

    return bag_mask


def crop_location_background(image: np.ndarray) -> np.ndarray:
    """Extract the background region above and beside the DoorDash bag for CLIP.

    Strategy: use the bag mask to determine where the bag is, then crop the
    largest contiguous background region ABOVE the bag's top edge. This gives
    CLIP a clean, artifact-free view of just the background (buildings, walls,
    sky) without any gray fills or compositing artifacts.

    The key insight: CLIP embeddings are highly sensitive to unnatural image
    artifacts (gray rectangles, composited strips). Feeding it a smaller but
    PURE background crop produces far better similarity scores than a full
    image with a gray-filled hole.
    """
    image = _ensure_rgb(image)
    h, w = image.shape[:2]
    if h < 4 or w < 4:
        return image

    bag_mask = _detect_bag_mask(image)
    mask_pixels = np.count_nonzero(bag_mask)

    if mask_pixels >= h * w * 0.02:
        # Find the top edge of the bag — everything above it is pure background
        rows_with_bag = np.where(bag_mask.any(axis=1))[0]
        bag_top = rows_with_bag[0] if len(rows_with_bag) > 0 else int(h * 0.55)

        # Also find the horizontal extent of the bag to extract side strips
        cols_with_bag = np.where(bag_mask.any(axis=0))[0]
        bag_left = cols_with_bag[0] if len(cols_with_bag) > 0 else int(w * 0.30)
        bag_right = cols_with_bag[-1] if len(cols_with_bag) > 0 else int(w * 0.70)
    else:
        # No bag detected — assume center-bottom placement
        bag_top = int(h * 0.55)
        bag_left = int(w * 0.30)
        bag_right = int(w * 0.70)

    # Primary crop: everything above the bag (full width) — this is where
    # buildings, walls, sky, and architectural details are most visible.
    # Take a bit more than just above the bag (add 10% overlap) to capture
    # context at the bag's level (e.g., wall texture beside the bag).
    crop_bottom = min(h, int(bag_top + h * 0.08))
    top_crop = image[:crop_bottom, :]

    # If the top crop is very narrow (bag is very high), include side strips
    if crop_bottom < h * 0.35:
        # Bag is positioned high — also grab left and right side columns
        left_strip = image[:, :max(1, bag_left)]
        right_strip = image[:, min(w - 1, bag_right):]
        # Stack: top full-width, then sides joined horizontally below
        sides = np.concatenate([left_strip, right_strip], axis=1)
        if sides.shape[1] != top_crop.shape[1]:
            sides = cv2.resize(
                sides, (top_crop.shape[1], max(1, sides.shape[0])),
                interpolation=cv2.INTER_AREA,
            )
        top_crop = np.concatenate([top_crop, sides], axis=0)

    # Ensure minimum size
    if top_crop.shape[0] < 32 or top_crop.shape[1] < 32:
        return image[:max(32, int(h * 0.5)), :]

    return top_crop


def enhance_for_matching(image: np.ndarray) -> np.ndarray:
    """Enhance image for CLIP matching with contrast normalization and detail sharpening."""
    # CLAHE on luminance for contrast normalization
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    # Mild unsharp mask to boost architectural edges (brick, window frames, signage)
    # without introducing noise artifacts
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(enhanced, 1.3, blurred, -0.3, 0)

    return sharpened


def resize_max_side(image: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
