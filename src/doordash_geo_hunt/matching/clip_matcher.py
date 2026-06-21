from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class MatchScore:
    lat: float
    lng: float
    score: float
    heading: float | None = None
    source_id: str | None = None


class ClipMatcher:
    """Embedding-based visual similarity (works well for architecture / storefronts)."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        # Model inference is serialized so a single shared instance can be used
        # safely from the parallel agent threads.
        self._infer_lock = threading.Lock()

    @torch.inference_mode()
    def embed(self, image: Image.Image) -> np.ndarray:
        with self._infer_lock:
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            feats = self.model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy()[0]

    def rank(
        self,
        query: Image.Image,
        candidates: list[dict],
        top_k: int = 15,
    ) -> list[MatchScore]:
        if not candidates:
            return []
        query_vec = self.embed(query)
        scored: list[MatchScore] = []
        for cand in candidates:
            image = cand["image"]
            vec = self.embed(image)
            score = float(np.dot(query_vec, vec))
            scored.append(
                MatchScore(
                    lat=cand["lat"],
                    lng=cand["lng"],
                    score=score,
                    heading=cand.get("heading"),
                    source_id=str(cand.get("pano_id") or cand.get("image_id") or ""),
                )
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


_SHARED_LOCK = threading.Lock()
_SHARED_MATCHER: ClipMatcher | None = None


def get_clip_matcher(model_name: str = "ViT-B-32", pretrained: str = "openai") -> ClipMatcher:
    """Return a process-wide shared ``ClipMatcher``.

    The model (and the heavy ``torch`` / ``torchvision`` / ``open_clip`` imports
    it triggers) is initialized exactly once. Creating it lazily across several
    worker threads simultaneously caused partially-initialized-module import
    errors, so callers should warm this up on the main thread before fanning out
    to the parallel agents.
    """
    global _SHARED_MATCHER
    with _SHARED_LOCK:
        if _SHARED_MATCHER is None:
            _SHARED_MATCHER = ClipMatcher(model_name=model_name, pretrained=pretrained)
    return _SHARED_MATCHER
