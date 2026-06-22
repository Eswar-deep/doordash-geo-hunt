from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; doordash-geo-hunt/1.0)",
    "Accept": "application/json",
}

_KNOWN_CITIES = [
    "New York City", "New York", "New Jersey", "Los Angeles", "San Francisco",
    "Miami", "Austin", "Dallas", "Atlanta", "Houston", "Chicago", "Seattle",
    "Denver", "Phoenix", "Boston", "Philadelphia", "Washington", "San Diego",
    "Las Vegas", "Nashville", "Kansas City", "San Jose", "Toronto", "Vancouver",
]


@dataclass
class TweetMedia:
    index: int
    url: str
    local_path: Path


@dataclass
class ParsedTweet:
    tweet_id: str
    text: str
    city_hint: str | None
    media: list[TweetMedia]
    raw: dict


def _fetch_from_api(api_url: str) -> dict:
    resp = httpx.get(api_url, headers=_HTTP_HEADERS, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") not in (None, 200) and "tweet" not in payload:
        raise RuntimeError(f"Tweet API error: {str(payload.get('message', payload))[:200]}")
    return payload


def fetch_tweet(tweet_url: str) -> ParsedTweet:
    match = re.search(r"/status/(\d+)", tweet_url)
    if not match:
        raise ValueError(f"Invalid tweet URL: {tweet_url}")
    tweet_id = match.group(1)
    handle_match = re.search(r"x\.com/([^/]+)/status", tweet_url) or re.search(
        r"twitter\.com/([^/]+)/status", tweet_url
    )
    handle = handle_match.group(1) if handle_match else "DoorDash"

    errors: list[str] = []
    tweet = None
    for api_url in (
        f"https://api.fxtwitter.com/{handle}/status/{tweet_id}",
        f"https://api.vxtwitter.com/{handle}/status/{tweet_id}",
    ):
        try:
            payload = _fetch_from_api(api_url)
            tweet = payload.get("tweet") or payload
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{api_url}: {exc}")
    if tweet is None:
        raise RuntimeError("Could not fetch tweet. " + " | ".join(errors))

    text = tweet.get("text", "")
    city_hint = _guess_city(text)

    media: list[TweetMedia] = []
    photos = tweet.get("media", {}).get("photos", [])
    if not photos and tweet.get("media_extended"):
        photos = [m for m in tweet["media_extended"] if m.get("type") == "photo"]
    for idx, photo in enumerate(photos, start=1):
        url = photo.get("url") or photo.get("media_url") or photo.get("media_url_https")
        if url:
            media.append(TweetMedia(index=idx, url=url, local_path=Path()))

    if len(media) < 4:
        raise RuntimeError(f"Expected 4 photos in tweet, got {len(media)}")

    return ParsedTweet(tweet_id=tweet_id, text=text, city_hint=city_hint, media=media, raw=tweet)


def _guess_city(text: str) -> str | None:
    if not text:
        return None
    # "New Jersey!! ..." or "NEW YORK CITY!!" — leading shout before punctuation.
    lead = re.match(r"\s*([A-Za-z][A-Za-z .'-]{2,30}?)\s*[!:]{1,3}", text)
    candidates: list[str] = []
    if lead:
        candidates.append(lead.group(1).strip())
    # "City, ST"
    m = re.search(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?),\s*[A-Z]{2}\b", text)
    if m:
        candidates.append(m.group(1).strip())
    # "in Dallas"
    m = re.search(r"\bin\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)\b", text)
    if m:
        candidates.append(m.group(1).strip())
    # "#Dallas"
    m = re.search(r"#([A-Za-z]{3,})", text)
    if m:
        candidates.append(m.group(1).strip())

    upper = text.upper()
    for cand in candidates:
        for known in _KNOWN_CITIES:
            if known.upper() == cand.upper():
                return known
    for known in _KNOWN_CITIES:
        if known.upper() in upper:
            return known
    # Fall back to the first reasonable leading candidate.
    for cand in candidates:
        if 2 < len(cand) <= 30:
            return cand.title()
    return None


def _download_one(item: TweetMedia, output_dir: Path, attempts: int = 3) -> TweetMedia:
    path = output_dir / f"photo{item.index}.jpg"
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = httpx.get(item.url, headers=_HTTP_HEADERS, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            return TweetMedia(index=item.index, url=item.url, local_path=path)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Failed to download photo{item.index} after {attempts} tries: {last_exc}")


def download_tweet_media(tweet: ParsedTweet, output_dir: Path, workers: int = 4) -> ParsedTweet:
    output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        downloaded = list(pool.map(lambda m: _download_one(m, output_dir), tweet.media))
    tweet.media = sorted(downloaded, key=lambda m: m.index)
    (output_dir / "tweet.json").write_text(json.dumps(tweet.raw, indent=2), encoding="utf-8")
    return tweet


def _dims(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:  # noqa: BLE001
        return (0, 0)


def _map_score(path: Path) -> float:
    """Score how likely a photo is the MAP (red zone on a map background).

    A real map has: (1) significant red/warm area AND (2) non-red regions with
    map-like muted tones (gray/green/white streets). Pure red promo graphics
    score lower because their non-red area is small or also saturated.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:  # noqa: BLE001
        return 0.0
    arr = np.asarray(img.resize((128, 128)), dtype=np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    warm = (r > g + 25) & (r > b + 25)
    warm_frac = float(warm.mean())
    # Map backgrounds have a mix of warm zone + muted map tiles. A pure red
    # promo graphic has warm_frac > 0.5 (mostly solid red/warm). The map sweet
    # spot is 0.05–0.45 (circular zone on a neutral map).
    if warm_frac > 0.55:
        return warm_frac * 0.4  # penalize — likely a full-bleed promo graphic
    return warm_frac


def classify_photos(output_dir: Path) -> dict:
    """Identify map (warm/red circle) vs location clue among the 4 photos.

    DoorDash Seat Drops follow a consistent 4-photo grid:
      photo1 = promo (ticket tag)
      photo2 = MAP (warm zone circle on a map)
      photo3 = LOCATION CLUE (bag on pedestal with background)
      photo4 = promo (GO FIND THEM)

    The classifier defaults to this canonical order and only overrides photo2 as
    the map if another photo scores substantially higher on the map heuristic AND
    photo2 scores very low (i.e. clearly not a map).
    """
    photos = [output_dir / f"photo{i}.jpg" for i in range(1, 5)]
    for p in photos:
        if not p.exists():
            raise FileNotFoundError(f"Expected {p}")

    scores = {p: _map_score(p) for p in photos}
    canonical_map = output_dir / "photo2.jpg"
    canonical_loc = output_dir / "photo3.jpg"

    best_map = max(photos, key=lambda p: scores[p])
    best_score = scores[best_map]
    canonical_score = scores[canonical_map]

    # Only override canonical if:
    #   (a) canonical photo2 has almost no red (< 0.03), AND
    #   (b) the best alternative scores significantly higher (> 2x).
    # Otherwise trust the fixed grid order — DoorDash hasn't changed it.
    if canonical_score < 0.03 and best_score > canonical_score * 2 and best_map != canonical_map:
        map_path = best_map
        confidence = round(min(0.9, 0.4 + best_score), 3)
    else:
        map_path = canonical_map
        confidence = round(min(0.95, 0.5 + canonical_score), 3) if canonical_score > 0.03 else 0.5

    # Location is always photo3 unless photo3 IS the map (extremely unlikely).
    if canonical_loc == map_path:
        loc_path = max(
            [p for p in photos if p != map_path],
            key=lambda p: _dims(p)[0] * _dims(p)[1],
        )
    else:
        loc_path = canonical_loc

    lw, lh = _dims(loc_path)
    if lw < 400 or lh < 400:
        fallback = canonical_loc if canonical_loc != map_path else output_dir / "photo2.jpg"
        if _dims(fallback)[0] >= 400:
            loc_path = fallback

    return {
        "promo1": output_dir / "photo1.jpg",
        "map": map_path,
        "location": loc_path,
        "promo2": output_dir / "photo4.jpg",
        "classification_confidence": confidence,
    }


def ingest_tweet(
    tweet_url: str,
    output_dir: Path,
    workers: int = 4,
    city_override: str | None = None,
) -> dict:
    tweet = fetch_tweet(tweet_url)
    tweet = download_tweet_media(tweet, output_dir, workers=workers)
    classified = classify_photos(output_dir)
    confidence = classified.pop("classification_confidence")
    city_hint = city_override or tweet.city_hint

    manifest = {
        "tweet_id": tweet.tweet_id,
        "url": tweet_url,
        "tweet_text": tweet.text,
        "city_hint": city_hint,
        "photo_classification": {k: str(v) for k, v in classified.items()},
        "classification_confidence": confidence,
        "files": {k: str(v) for k, v in classified.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "city_hint": city_hint or "",
        "tweet_id": tweet.tweet_id,
        "tweet_text": tweet.text,
        "classification_confidence": confidence,
        **classified,
    }
