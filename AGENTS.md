# AGENTS.md

## Cursor Cloud specific instructions

### What this is
`doordash-geo-hunt` is a single **Python CLI** package (no web server, no database, no
frontend). The entry point is `python cli.py` (subcommands `ingest` and `run`); see
`README.md` for the full command reference. Dependencies are installed by the startup
update script (`pip install -e .`), so you should not need to install anything manually.

### Running the pipeline
- Use a manual circle to skip the map vision-LLM call:
  `python cli.py run --map <map.jpg> --location <loc.jpg> --center-lat .. --center-lng .. --radius-m .. --city Miami --output-json output/x.json`
- Bundled offline inputs live in `samples/miami-drop1/` and `samples/miami-drop2/`.
- First run downloads model weights from the network (open_clip `ViT-B-32` and EasyOCR
  detection/recognition models, a few hundred MB). Internet access is required.

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
- With **no** keys configured, all 5 agents fail gracefully and the judge raises
  `No in-circle candidates from any agent`, so the pipeline exits non-zero. This is the
  expected keyless behavior, not a bug. `python scripts/test_apis.py` smoke-tests whichever
  keys are present.

### Known caveats in the visual-matcher path
- The three CLIP visual matchers (`streetview`, `mapillary`, `kartaview`) currently raise
  before producing candidates even with keys: `preprocessing.crop_location_background`
  concatenates mismatched array shapes, and `torchvision` can also hit an import race when
  the 5 agents run in parallel. In practice, candidates therefore come from the VLM agents
  (`vlm_geoguesser`, `landmark_ocr`). The orchestrator catches these per-agent errors, so
  they degrade gracefully rather than crashing the run.

### Lint / test / build
- There is no test suite, linter config, or build step in this repo. "Build" is just the
  editable install; "run" is the CLI above.
