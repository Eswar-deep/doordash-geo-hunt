#!/usr/bin/env python3
"""CLI — ingest tweets and run the geo-hunt pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from doordash_geo_hunt.models import ContestInput, FinalVerdict, SearchRegion
from doordash_geo_hunt.orchestrator import (
    AgentTimeouts,
    PipelineConfig,
    format_report,
    run_contest,
    save_json_output,
)
from doordash_geo_hunt.pipeline_context import StreetViewConfig
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

    # Agent selection
    parser.add_argument(
        "--agents",
        default="streetview,vlm",
        help="Comma list: streetview,vlm,landmark,mapillary,kartaview (default: streetview,vlm)",
    )

    # Staged output
    parser.add_argument("--staged", dest="staged", action="store_true", default=True)
    parser.add_argument("--no-staged", dest="staged", action="store_false")
    parser.add_argument("--staged-parallel", dest="staged_parallel", action="store_true", default=True)
    parser.add_argument("--no-staged-parallel", dest="staged_parallel", action="store_false")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None, help="Debug single stage")

    # Parallelism knobs
    import os

    cpu = os.cpu_count() or 8
    parser.add_argument("--sv-workers", type=int, default=min(64, cpu * 4))
    parser.add_argument("--clip-batch-size", type=int, default=256)
    parser.add_argument("--ingest-workers", type=int, default=4)
    parser.add_argument("--judge-workers", type=int, default=4)

    # Street View coarse-to-fine
    parser.add_argument("--sv-coarse-fine", dest="sv_coarse_fine", action="store_true", default=True)
    parser.add_argument("--no-sv-coarse-fine", dest="sv_coarse_fine", action="store_false")
    parser.add_argument("--sv-headings-coarse", type=int, default=8)
    parser.add_argument("--sv-headings-fine", type=int, default=12)
    parser.add_argument("--sv-headings", type=int, default=None, help="Override heading count (coarse-fine off)")
    parser.add_argument("--sv-heading-step", type=int, default=None, help="Derive heading count from degree step")
    parser.add_argument("--sv-refine-headings", dest="sv_refine", action="store_true", default=True)
    parser.add_argument("--no-sv-refine-headings", dest="sv_refine", action="store_false")
    parser.add_argument("--sv-refine-span", type=int, default=45)
    parser.add_argument("--sv-refine-step", type=int, default=3)
    parser.add_argument("--sv-pitch-refine", default="0,10,20,30,40,-10,-20")
    parser.add_argument("--sv-max-frames", type=int, default=20000)
    parser.add_argument("--sv-step-m", type=float, default=None)
    parser.add_argument("--sv-max-panos", type=int, default=1500)
    parser.add_argument("--sv-cache", action="store_true", default=False, help="Dev only: cache SV frames")

    # Per-agent timeouts
    parser.add_argument("--agent-timeout-streetview", type=float, default=900.0)
    parser.add_argument("--agent-timeout-vlm", type=float, default=90.0)
    parser.add_argument("--agent-timeout-landmark", type=float, default=120.0)


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    agents = [a.strip() for a in str(args.agents).split(",") if a.strip()]
    pitches = tuple(
        float(p) for p in str(args.sv_pitch_refine).split(",") if p.strip() != ""
    ) or (0.0,)
    sv = StreetViewConfig(
        coarse_fine=args.sv_coarse_fine,
        headings_coarse=args.sv_headings_coarse,
        headings_fine=args.sv_headings_fine,
        headings_override=args.sv_headings,
        heading_step=args.sv_heading_step,
        refine_headings=args.sv_refine,
        refine_span=args.sv_refine_span,
        refine_step=args.sv_refine_step,
        pitch_refine=pitches,
        max_frames=args.sv_max_frames,
        max_panos=args.sv_max_panos,
        step_m=args.sv_step_m,
        workers=args.sv_workers,
        clip_batch_size=args.clip_batch_size,
        cache=args.sv_cache,
    )
    timeouts = AgentTimeouts(
        streetview=args.agent_timeout_streetview,
        vlm=args.agent_timeout_vlm,
        landmark=args.agent_timeout_landmark,
    )

    staged = args.staged
    staged_parallel = args.staged_parallel
    run_judge = True
    if args.stage == 1:
        agents = ["vlm"]
        run_judge = False
    elif args.stage == 2:
        agents = [a for a in agents if a in ("streetview", "vlm")] or ["streetview", "vlm"]
        run_judge = False

    return PipelineConfig(
        agents=agents,
        staged=staged,
        staged_parallel=staged_parallel,
        sv=sv,
        timeouts=timeouts,
        judge_workers=args.judge_workers,
        cache_dir=args.cache_dir,
        run_judge=run_judge,
    )


_STAGE_TITLES = {
    "p1": "P1 VERDICT",
    "p2": "P2 VERDICT",
    "p3": "P3 VERDICT",
    "p3_densify": "P3 DENSIFY VERDICT",
    "p3_final": "P3 FINAL VERDICT",
    "p4_final": "P4 FINAL VERDICT",
    "p1_fast": "P1 FAST VERDICT",
    "p2_clip": "P2 CLIP VERDICT",
}


def _print_stage(stage: str, verdict: FinalVerdict) -> None:
    winner = verdict.winner_agent.value if verdict.winner_agent else "ensemble"
    title = _STAGE_TITLES.get(stage, f"{stage.upper()} VERDICT")
    prov = " (PROVISIONAL)" if verdict.provisional else ""
    print(f"\n=== {title}{prov} ===")
    print(
        f"lat: {verdict.lat:.6f}  lng: {verdict.lng:.6f}  "
        f"confidence: {verdict.confidence:.3f}  agent: {winner}"
    )
    if not verdict.provisional:
        if verdict.street_view_url:
            print(f"Street View: {verdict.street_view_url}")
        print(f"Reasoning: {verdict.reasoning}")
    print(f"Google Maps: {verdict.maps_url()}")


def _stage_path(final_path: Path, stage: str) -> Path:
    return final_path.with_name(f"{final_path.stem}_{stage}.json")


def _make_stage_emitter(final_path: Path):
    def on_stage(stage, region, verdict, agent_results):
        _print_stage(stage, verdict)
        save_json_output(_stage_path(final_path, stage), region, agent_results, verdict)
        if stage in ("p3_final", "p4_final"):
            save_json_output(final_path, region, agent_results, verdict)

    return on_stage


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
    cfg = _build_config(args)
    final_path = args.output_json

    try:
        if cfg.staged:
            emitter = _make_stage_emitter(final_path)
            region, agent_results, verdict = run_contest(contest, cfg, on_stage=emitter)
        else:
            region, agent_results, verdict = run_contest(contest, cfg)
            print(format_report(region, agent_results, verdict))
            save_json_output(final_path, region, agent_results, verdict)
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nSaved JSON: {final_path}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    try:
        result = ingest_tweet(args.url, args.out, workers=args.ingest_workers, city_override=args.city)
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
    args.map = Path(result["map"])
    args.location = Path(result["location"])
    args.city = args.city or (city_hint if city_hint else None)
    args.output_json = output_json
    return _run_pipeline(args)


def _cmd_prewarm(args: argparse.Namespace) -> int:
    """Load model weights (CLIP + torch + LoFTR, optionally EasyOCR). NOT Street View images."""
    print("Prewarming models (CLIP + LoFTR + torch)...")
    try:
        from doordash_geo_hunt.matching.clip_matcher import get_clip_matcher

        get_clip_matcher()
        print("  CLIP ready.")
    except Exception as exc:  # noqa: BLE001
        print(f"  CLIP failed: {exc}", file=sys.stderr)
        return 1
    try:
        from doordash_geo_hunt.matching.feature_matcher import _get_matcher

        _get_matcher()
        print("  LoFTR ready.")
    except Exception as exc:  # noqa: BLE001
        print(f"  LoFTR failed: {exc}", file=sys.stderr)
        return 1
    if args.ocr:
        try:
            from doordash_geo_hunt.agents.vlm_agents import get_ocr_reader

            get_ocr_reader()
            print("  EasyOCR ready.")
        except Exception as exc:  # noqa: BLE001
            print(f"  EasyOCR failed: {exc}", file=sys.stderr)
    print("Prewarm complete. (Street View images are NOT pre-cached — circle changes per drop.)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DoorDash FIFA geo-hunt — ingest tweets and locate the drop.",
    )
    sub = parser.add_subparsers(dest="command")

    ingest_parser = sub.add_parser("ingest", help="Download photos from a DoorDash tweet URL")
    ingest_parser.add_argument("url", help="Tweet URL (x.com/DoorDash/status/...)")
    ingest_parser.add_argument("--out", type=Path, default=Path("samples/live-drop"))
    ingest_parser.add_argument("--run", action="store_true", help="Run full pipeline after ingest")
    ingest_parser.add_argument("--tweet-id", action="store_true", help="Name JSON output by tweet id")
    _add_run_args(ingest_parser)
    ingest_parser.set_defaults(func=_cmd_ingest)

    run_parser = sub.add_parser("run", help="Run pipeline on map + location photos")
    _add_run_args(run_parser)
    run_parser.set_defaults(func=_run_pipeline)

    prewarm_parser = sub.add_parser("prewarm", help="Load CLIP/torch weights (not SV images)")
    prewarm_parser.add_argument("--ocr", action="store_true", help="Also warm EasyOCR")
    prewarm_parser.set_defaults(func=_cmd_prewarm)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    argv = argv if argv is not None else sys.argv[1:]

    # Legacy: python cli.py --map ... --location ...
    if argv and argv[0] not in ("ingest", "run", "prewarm") and any(
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
