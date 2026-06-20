from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; doordash-geo-hunt/1.0)",
    "Accept": "application/json",
}


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
    resp = httpx.get(
        api_url,
        headers=_HTTP_HEADERS,
        timeout=30,
        follow_redirects=True,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") not in (None, 200) and "tweet" not in payload:
        raise RuntimeError(f"Tweet API error: {payload.get('message', payload)[:200]}")
    return payload


def fetch_tweet(tweet_url: str) -> ParsedTweet:
    match = re.search(r"/status/(\d+)", tweet_url)
    if not match:
        raise ValueError(f"Invalid tweet URL: {tweet_url}")
    tweet_id = match.group(1)
    handle_match = re.search(r"x\.com/([^/]+)/status", tweet_url)
    handle = handle_match.group(1) if handle_match else "DoorDash"

    errors: list[str] = []
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
            tweet = None
    else:
        raise RuntimeError("Could not fetch tweet. " + " | ".join(errors))

    text = tweet.get("text", "")
    city_hint = _guess_city(text)

    media: list[TweetMedia] = []
    photos = tweet.get("media", {}).get("photos", [])
    if not photos and tweet.get("media_extended"):
        photos = [m for m in tweet["media_extended"] if m.get("type") == "photo"]
    for idx, photo in enumerate(photos, start=1):
        url = photo.get("url") or photo.get("media_url") or photo.get("media_url_https")
        if not url:
            continue
        media.append(TweetMedia(index=idx, url=url, local_path=Path()))

    if len(media) < 4:
        raise RuntimeError(f"Expected 4 photos in tweet, got {len(media)}")

    return ParsedTweet(
        tweet_id=tweet_id,
        text=text,
        city_hint=city_hint,
        media=media,
        raw=tweet,
    )


def _guess_city(text: str) -> str | None:
    upper = text.upper()
    for city in (
        "MIAMI",
        "AUSTIN",
        "DALLAS",
        "ATLANTA",
        "HOUSTON",
        "LOS ANGELES",
        "NEW YORK",
        "SAN FRANCISCO",
        "CHICAGO",
        "SEATTLE",
        "DENVER",
        "PHOENIX",
    ):
        if city in upper:
            return city.title()
    return None


def download_tweet_media(tweet: ParsedTweet, output_dir: Path) -> ParsedTweet:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[TweetMedia] = []
    for item in tweet.media:
        path = output_dir / f"photo{item.index}.jpg"
        resp = httpx.get(
            item.url,
            headers=_HTTP_HEADERS,
            timeout=60,
            follow_redirects=True,
        )
        resp.raise_for_status()
        path.write_bytes(resp.content)
        downloaded.append(TweetMedia(index=item.index, url=item.url, local_path=path))
    tweet.media = downloaded
    (output_dir / "tweet.json").write_text(json.dumps(tweet.raw, indent=2), encoding="utf-8")
    return tweet


def classify_photos(output_dir: Path) -> dict[str, Path]:
    """
    Heuristic ordering for DoorDash Seat Drops (4-photo grid):
      photo1 = promo (ticket tag)
      photo2 = map warm zone
      photo3 = location clue (bag on pedestal)
      photo4 = promo (GO FIND THEM)
    """
    mapping = {
        "promo1": output_dir / "photo1.jpg",
        "map": output_dir / "photo2.jpg",
        "location": output_dir / "photo3.jpg",
        "promo2": output_dir / "photo4.jpg",
    }
    for key, path in mapping.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected {path} for {key}")
    return mapping


def ingest_tweet(tweet_url: str, output_dir: Path) -> dict[str, Path | str]:
    tweet = fetch_tweet(tweet_url)
    tweet = download_tweet_media(tweet, output_dir)
    classified = classify_photos(output_dir)
    manifest = {
        "tweet_id": tweet.tweet_id,
        "url": tweet_url,
        "text": tweet.text,
        "city_hint": tweet.city_hint,
        "files": {k: str(v) for k, v in classified.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"city_hint": tweet.city_hint or "", "tweet_id": tweet.tweet_id, **classified}
