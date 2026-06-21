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


def crop_location_background(image: np.ndarray) -> np.ndarray:
    """
    DoorDash location photos often have a bag/pedestal in the center-bottom.
    Keep top band + side strips where background is most visible, then compose
    them into a single image for CLIP matching.

    The side strips span the full height, so they are joined horizontally first
    and then stacked under the top band (widths are matched via resize).
    """
    h, w = image.shape[:2]
    top = image[: int(h * 0.45)]
    left = image[:, : int(w * 0.22)]
    right = image[:, int(w * 0.78) :]

    sides = np.concatenate([left, right], axis=1)
    if sides.shape[1] != top.shape[1]:
        sides = cv2.resize(
            sides,
            (top.shape[1], sides.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return np.concatenate([top, sides], axis=0)


def enhance_for_matching(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def resize_max_side(image: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
