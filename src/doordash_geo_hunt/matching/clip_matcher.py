from __future__ import annotations

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

    @torch.inference_mode()
    def embed(self, image: Image.Image) -> np.ndarray:
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
