from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

# ViT-H-14 — best fine-grained discrimination. Needs GPU for 3000+ frames.
# On CPU, reduce --sv-max-frames or set CLIP_MODEL_NAME=ViT-L-14.
_DEFAULT_MODEL = os.getenv("CLIP_MODEL_NAME", "ViT-H-14")
_DEFAULT_PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s32b_b79k")


@dataclass
class MatchScore:
    lat: float
    lng: float
    score: float
    heading: float | None = None
    source_id: str | None = None


class ClipMatcher:
    """Embedding-based visual similarity (works well for architecture / storefronts)."""

    def __init__(self, model_name: str | None = None, pretrained: str | None = None) -> None:
        import open_clip

        model_name = model_name or _DEFAULT_MODEL
        pretrained = pretrained or _DEFAULT_PRETRAINED
        print(f"[clip] Loading {model_name} ({pretrained})...", file=sys.stderr, flush=True)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        self._infer_lock = threading.Lock()
        self._embed_dim = self.model.visual.output_dim
        print(f"[clip] Ready: {model_name} dim={self._embed_dim} device={self.device}", file=sys.stderr, flush=True)

    @torch.inference_mode()
    def embed(self, image: Image.Image) -> np.ndarray:
        with self._infer_lock:
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            feats = self.model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy()[0]

    @torch.inference_mode()
    def embed_batch(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Embed many images in GPU/CPU batches. Returns (N, D) L2-normalized array."""
        if not images:
            return np.empty((0, self._embed_dim), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            with self._infer_lock:
                tensors = torch.stack([self.preprocess(img) for img in chunk]).to(self.device)
                feats = self.model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                vectors.append(feats.cpu().numpy())
        return np.concatenate(vectors, axis=0)

    def rank_batched(
        self,
        query_vec: np.ndarray,
        candidates: list[dict],
        top_k: int = 15,
        batch_size: int = 32,
    ) -> list[MatchScore]:
        """Rank candidates against a precomputed query vector using batched embeds.

        PIL images are released from each candidate dict after embedding to keep
        memory bounded on large Street View sweeps.
        """
        if not candidates:
            return []
        images = [c["image"] for c in candidates]
        mat = self.embed_batch(images, batch_size=batch_size)
        sims = mat @ query_vec
        scored: list[MatchScore] = []
        for cand, score in zip(candidates, sims):
            cand.pop("image", None)  # free PIL image
            scored.append(
                MatchScore(
                    lat=cand["lat"],
                    lng=cand["lng"],
                    score=float(score),
                    heading=cand.get("heading"),
                    source_id=str(cand.get("pano_id") or cand.get("image_id") or ""),
                )
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def rank(
        self,
        query: Image.Image,
        candidates: list[dict],
        top_k: int = 15,
    ) -> list[MatchScore]:
        if not candidates:
            return []
        query_vec = self.embed(query)
        return self.rank_batched(query_vec, candidates, top_k=top_k)


_SHARED_LOCK = threading.Lock()
_SHARED_MATCHER: ClipMatcher | None = None


def get_clip_matcher(model_name: str | None = None, pretrained: str | None = None) -> ClipMatcher:
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
