#!/bin/bash
# Run this on a fresh g4dn.xlarge (Deep Learning AMI Ubuntu) to set up the pipeline.
set -e

echo "=== Setting up doordash-geo-hunt on GPU instance ==="

# Clone repo
cd ~
if [ -d doordash-geo-hunt ]; then
    cd doordash-geo-hunt && git pull
else
    git clone https://github.com/Eswar-deep/doordash-geo-hunt.git
    cd doordash-geo-hunt
fi

# Use system python (Deep Learning AMI has torch + CUDA pre-installed)
pip install -e . 2>&1 | tail -3

# Create .env from template (user must fill in keys)
if [ ! -f .env ]; then
    cat > .env << 'ENVEOF'
VISION_LLM_PROVIDER=bedrock
GOOGLE_MAPS_API_KEY=FILL_ME
AWS_BEARER_TOKEN_BEDROCK=FILL_ME
AWS_BEDROCK_REGION=us-east-1
AWS_BEDROCK_SONNET_MODEL_ID=us.anthropic.claude-sonnet-4-6
AWS_BEDROCK_OPUS_MODEL_ID=us.anthropic.claude-opus-4-6
AWS_BEDROCK_OPUS_FALLBACK_MODEL_ID=us.anthropic.claude-opus-4-5-20251101-v1:0

# ViT-H-14 on GPU (default — do not change on GPU instance)
# CLIP_MODEL_NAME=ViT-H-14
# CLIP_PRETRAINED=laion2b_s32b_b79k
ENVEOF
    echo ""
    echo ">>> IMPORTANT: Edit .env with your API keys:"
    echo ">>>   nano .env"
    echo ""
fi

# Prewarm — downloads ViT-H-14 weights (~3.9GB, one-time)
echo "=== Prewarming CLIP (ViT-H-14) + torch ==="
python cli.py prewarm

# Verify GPU detected
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=== READY ==="
echo "Run a drop:"
echo "  python cli.py ingest \"https://x.com/DoorDash/status/TWEET_ID\" --out samples/live-drop --run --tweet-id --agents streetview,vlm --staged --staged-parallel --sv-workers 32"
