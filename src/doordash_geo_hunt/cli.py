#!/usr/bin/env python3
"""CLI — ingest tweets and run the geo-hunt pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from doordash_geo_hunt.models import ContestInput, SearchRegion
from doordash_geo_hunt.orchestrator import format_report, run_pipeline_sync, save_json_output
from doordash_geo_hunt.twitter_fetcher import ingest_tweet


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--map", type=Path, help="Map screenshot with circle")
    parser.add_argument("--location", type=Path, help="On-site location photo")
    parser.add_argument("--city", default=None, help="City hint (e.g. Miami)")
    parser.add_argument("--center-lat", type=float, default=None, help="Manual circle center lat")
    parser.add_argument("--center-lng", type=float, default=None, help="Manual circle center lng")
    parser.add_argument("--radius-m", type=float, default=None, help="Manual circle radius meters")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--output-json", type=Path, default=Path("output/result.json"))


def _run_pipeline(args: argparse.Namespace) -> int:
    if args.map is None or args.location is None:
        print("Pipeline requires --map and --location.", file=sys.stderr)
        return 2

    region_override = None
    if args.center_lat is not None and args.center_lng is not None and args.radius_m is not None:
        region_override = SearchRegion(
            center_lat=args.center_lat,
            center_lng=args.center_lng,
            radius_m=args.radius_m,
            city=args.city,
            source="manual_cli",
        )

    contest = ContestInput(
        map_image=args.map,
        location_image=args.location,
        city_hint=args.city,
        region_override=region_override,
    )

    try:
        region, agent_results, verdict = run_pipeline_sync(contest, cache_dir=args.cache_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    print(format_report(region, agent_results, verdict))
    save_json_output(args.output_json, region, agent_results, verdict)
    print(f"\nSaved JSON: {args.output_json}")
    if verdict.street_view_url:
        print(f"Maps: https://www.google.com/maps?q={verdict.lat},{verdict.lng}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    try:
        result = ingest_tweet(args.url, args.out)
    except Exception as exc:  # noqa: BLE001
        print(f"Ingest failed: {exc}", file=sys.stderr)
        return 1

    city_hint = str(result.get("city_hint") or "")
    tweet_id = str(result.get("tweet_id", args.out.name))
    print(f"Ingested tweet {tweet_id} -> {args.out}")
    print(f"  map:      {result['map']}")
    print(f"  location: {result['location']}")
    if city_hint:
        print(f"  city:     {city_hint}")

    if not args.run:
        print('\nRun pipeline: python cli.py ingest "<url>" --out samples/live-drop --run')
        return 0

    output_json = Path("output") / f"{tweet_id}.json" if args.tweet_id else args.output_json
    run_args = argparse.Namespace(
        map=Path(result["map"]),
        location=Path(result["location"]),
        city=args.city or (city_hint if city_hint else None),
        center_lat=args.center_lat,
        center_lng=args.center_lng,
        radius_m=args.radius_m,
        cache_dir=args.cache_dir,
        output_json=output_json,
    )
    return _run_pipeline(run_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DoorDash FIFA geo-hunt — ingest tweets and locate the drop.",
    )
    sub = parser.add_subparsers(dest="command")

    ingest_parser = sub.add_parser(
        "ingest",
        help="Download photos from a DoorDash tweet URL",
    )
    ingest_parser.add_argument("url", help="Tweet URL (x.com/DoorDash/status/...)")
    ingest_parser.add_argument(
        "--out",
        type=Path,
        default=Path("samples/live-drop"),
        help="Output folder for photos + manifest",
    )
    ingest_parser.add_argument(
        "--run",
        action="store_true",
        help="Run full pipeline after ingest",
    )
    ingest_parser.add_argument("--city", default=None, help="Override city hint from tweet")
    ingest_parser.add_argument("--center-lat", type=float, default=None)
    ingest_parser.add_argument("--center-lng", type=float, default=None)
    ingest_parser.add_argument("--radius-m", type=float, default=None)
    ingest_parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    ingest_parser.add_argument("--output-json", type=Path, default=Path("output/result.json"))
    ingest_parser.add_argument("--tweet-id", action="store_true", help="Name JSON output by tweet id")
    ingest_parser.set_defaults(func=_cmd_ingest)

    run_parser = sub.add_parser("run", help="Run pipeline on map + location photos")
    _add_run_args(run_parser)
    run_parser.set_defaults(func=_run_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    argv = argv if argv is not None else sys.argv[1:]

    # Legacy: python cli.py --map ... --location ...
    if argv and argv[0] != "ingest" and argv[0] != "run" and any(
        a in ("--map", "--location") for a in argv
    ):
        legacy = argparse.ArgumentParser(description="Locate DoorDash FIFA ticket drop.")
        _add_run_args(legacy)
        args = legacy.parse_args(argv)
        return _run_pipeline(args)

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
