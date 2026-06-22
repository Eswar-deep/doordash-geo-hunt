# AGENTS.md

## Cursor Cloud specific instructions

### What this is
`doordash-geo-hunt` is a single **Python CLI** package (no web server, no database, no
frontend). The entry point is `python cli.py` (subcommands `ingest`, `run`, `prewarm`);
see `README.md` for the full command reference. Dependencies are installed by the startup
update script (`pip install -e .`), so you should not need to install anything manually.

### Running the pipeline
- Default agent roster is `streetview,vlm` (Mapillary, KartaView, and landmark/OCR are
  opt-in via `--agents`). Runs are **staged + parallel** by default: VLM and Street View
  start together and the CLI auto-prints **P1 (fast) → P2 (CLIP) → P3 (final)** plus
  `output/<id>_p1_fast.json`, `_p2_clip.json`, `_p3_final.json`, and `<id>.json`. No user
  interaction is needed mid-run.
- Use a manual circle to skip the map vision-LLM call:
  `python cli.py run --map <map.jpg> --location <loc.jpg> --center-lat .. --center-lng .. --radius-m .. --city Miami --output-json output/x.json`
- `python cli.py prewarm` loads CLIP/torch weights only (NOT Street View images — the
  drop circle changes daily and is unknown until the tweet ingests).
- Bundled offline inputs live in `samples/miami-drop1/` and `samples/miami-drop2/`.
- First run downloads model weights from the network (open_clip `ViT-B-32` and, only when
  `--agents landmark` is used, EasyOCR models). Internet access is required.
- `--sv-cache` writes Street View frames to disk and is **dev-only**; contest runs fetch
  fresh frames (no disk cache).

### API keys are required for a real verdict (non-obvious)
- The pipeline only produces a final verdict if at least one **vision-LLM provider** is
  configured (`GEMINI_API_KEY`, AWS Bedrock via `AWS_BEARER_TOKEN_BEDROCK`, Anthropic,
  OpenAI, or Azure OpenAI). Optional: `GOOGLE_MAPS_API_KEY` and `MAPILLARY_ACCESS_TOKEN`
  for the visual-matcher agents.
- Configure keys via repo `.env` (copy `.env.example`) or as environment variables; the
  CLI loads `.env` via `python-dotenv` (it does NOT override already-set env vars).
- **Provider-selection gotcha:** when `VISION_LLM_PROVIDER` is unset, `llm_vision.active_vision_provider()`
  auto-picks the first provider whose key is present, checking `GEMINI_API_KEY` **before**
  Bedrock. The cloud secrets in this environment include a `GEMINI_API_KEY` whose billing
  is depleted (every Gemini model returns `429 "prepayment credits are depleted"`). To use
  the working Bedrock key you MUST set `VISION_LLM_PROVIDER=bedrock` — e.g. in a local
  `.env` (gitignored, so recreate it as needed) or as an env var. The default Bedrock Opus
  model id is invalid on this account, but the code auto-falls back to the configured Opus
  fallback (`AWS_BEDROCK_OPUS_FALLBACK_MODEL_ID`), which works.
- With **no** keys configured the agents fail gracefully and the judge raises
  `No in-circle candidates from any agent`, so the pipeline exits non-zero. This is the
  expected keyless behavior, not a bug. A fast **preflight** also aborts in <5s if the
  `streetview` agent is enabled without `GOOGLE_MAPS_API_KEY`, or if no vision provider is
  configured and no manual circle is given. `python scripts/test_apis.py` smoke-tests keys.

### Visual-matcher path (fixed)
- `preprocessing.crop_location_background` now coerces RGBA/grayscale to RGB and guards
  tiny images; the CLIP model is a warmed shared singleton, so the earlier shape-mismatch
  and torch/open_clip import races no longer occur. Street View runs a 3-phase parallel
  coarse→fine→refine sweep (shared connection-pooled `httpx.Client`, pano dedup, batched
  CLIP). Mapillary/KartaView remain flaky and are off by default.

### Lint / test / build
- Tests live in `tests/` (pytest). Install dev deps with `pip install -e ".[dev]"` and run
  `pytest -q`. There is no linter config; "build" is just the editable install.
