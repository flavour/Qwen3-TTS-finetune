#!/usr/bin/env bash
# Qwen3-TTS Setup Script (uv-based, no WhisperX)
#
# Usage:
#   ./setup.sh          # Interactive mode (prompts for model download)
#   ./setup.sh --auto   # Non-interactive mode (skips model download prompt)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

AUTO_MODE=false
[[ "$1" == "--auto" ]] && AUTO_MODE=true

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Qwen3-TTS Setup Script (uv)${NC}"
$AUTO_MODE && echo -e "${YELLOW}(Running in auto mode)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check for uv
echo -e "${GREEN}Checking for uv...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv not found${NC}"
    echo -e "${YELLOW}Install: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi
echo -e "${GREEN}Found: $(uv --version)${NC}"
echo ""

# Check for sox
echo -e "${GREEN}Checking for SoX...${NC}"
if ! command -v sox &> /dev/null; then
    echo -e "${YELLOW}Warning: SoX not found (required by qwen-tts)${NC}"
    echo -e "${YELLOW}Install: sudo apt install sox libsox-fmt-all${NC}"
else
    echo -e "${GREEN}Found: $(sox --version 2>&1 | head -1)${NC}"
fi
echo ""

# Detect CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo -e "${GREEN}Detected CUDA: $CUDA_VERSION${NC}"
else
    echo -e "${YELLOW}No CUDA detected.${NC}"
fi
echo ""

# Create venv with uv
echo -e "${GREEN}Step 1: Creating virtual environment with uv${NC}"
if [ ! -d ".venv" ]; then
    uv venv --python 3.12
    echo -e "${GREEN}Virtual environment created at .venv${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists at .venv${NC}"
fi
echo ""

# Install dependencies
echo -e "${GREEN}Step 2: Installing dependencies${NC}"
uv pip install \
    torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

uv pip install \
    qwen-tts \
    numpy \
    librosa \
    soundfile \
    tqdm \
    transformers \
    accelerate \
    safetensors \
    datasets \
    huggingface-hub \
    hf_transfer \
    tensorboard

echo -e "${GREEN}Dependencies installed${NC}"
echo ""

# Try flash-attn
echo -e "${GREEN}Step 3: Installing flash-attn (optional)${NC}"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install "$FLASH_ATTN_WHEEL" 2>/dev/null && \
    echo -e "${GREEN}flash-attn installed${NC}" || \
    echo -e "${YELLOW}flash-attn failed — will use eager attention (slower but OK)${NC}"
echo ""

# Pre-download models
echo -e "${GREEN}Step 4: Pre-downloading models${NC}"
if [ "$AUTO_MODE" = true ]; then
    echo -e "${YELLOW}Auto mode: skipping model download (will download on first run)${NC}"
else
    read -p "Pre-download models now? Saves time during training. (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Downloading Qwen3-TTS Tokenizer...${NC}"
        uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-TTS-Tokenizer-12Hz', local_dir='./models/Qwen3-TTS-Tokenizer-12Hz')
" 2>/dev/null || echo "Tokenizer download will happen during first run"

        echo -e "${GREEN}Downloading Qwen3-TTS Base Model...${NC}"
        uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='./models/Qwen3-TTS-12Hz-1.7B-Base')
" 2>/dev/null || echo "Base model download will happen during first run"

        echo -e "${GREEN}Models downloaded to ./models/${NC}"
    else
        echo -e "${YELLOW}Skipping. Models will download on first run.${NC}"
    fi
fi
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "To train:"
echo -e "  ${YELLOW}./train.sh --audio_dir ./audio --ref_audio ./ref.wav --speaker_name my_voice${NC}"
echo -e ""
echo -e "Or with pre-built JSONL (skip transcription):"
echo -e "  ${YELLOW}./train.sh --jsonl ./train_raw.jsonl --ref_audio ./ref.wav --speaker_name my_voice${NC}"
echo ""
