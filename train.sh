#!/usr/bin/env bash
# Qwen3-TTS One-Command Fine-Tuning Script (uv-based, no WhisperX)
#
# Usage:
#   # With pre-built JSONL (our workflow — hand-corrected transcripts):
#   ./train.sh --jsonl ./train_raw.jsonl --ref_audio ./ref.wav --speaker_name lagertha
#
#   # With audio dir (auto-transcribes if no --jsonl):
#   ./train.sh --audio_dir ./audio --ref_audio ./ref.wav --speaker_name my_voice

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check system deps
check_system_deps() {
    if ! command -v sox &> /dev/null; then
        echo -e "${YELLOW}Installing sox...${NC}"
        if command -v apt-get &> /dev/null; then
            apt-get update -qq && apt-get install -y -qq sox libsox-fmt-all 2>/dev/null || {
                echo -e "${RED}Please install sox: sudo apt install sox libsox-fmt-all${NC}"
                exit 1
            }
        else
            echo -e "${RED}Please install sox manually${NC}"
            exit 1
        fi
    fi
}

check_system_deps

# Check environment
check_environment_ready() {
    [ -d "$SCRIPT_DIR/.venv" ] && \
    "$SCRIPT_DIR/.venv/bin/python" -c "import torch; import qwen_tts" 2>/dev/null
}

# Defaults
AUDIO_DIR=""
JSONL=""
REF_AUDIO=""
SPEAKER_NAME="my_speaker"
OUTPUT_DIR="./output"
DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
BATCH_SIZE=2
LR=2e-5
EPOCHS=3
LANGUAGE="en"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --audio_dir)     AUDIO_DIR="$2"; shift 2 ;;
        --jsonl)         JSONL="$2"; shift 2 ;;
        --ref_audio)     REF_AUDIO="$2"; shift 2 ;;
        --speaker_name)  SPEAKER_NAME="$2"; shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --batch_size)    BATCH_SIZE="$2"; shift 2 ;;
        --lr)            LR="$2"; shift 2 ;;
        --epochs)        EPOCHS="$2"; shift 2 ;;
        --language)      LANGUAGE="$2"; shift 2 ;;
        --lora)          LORA=true; shift ;;
        --lora_rank)     LORA_RANK="$2"; shift 2 ;;
        --gradient_checkpointing) GRAD_CKPT=true; shift ;;
        --warmup_steps)  WARMUP_STEPS="$2"; shift 2 ;;
        --early_stopping) EARLY_STOPPING="$2"; shift 2 ;;
        --mlflow_url)    MLFLOW_URL="$2"; shift 2 ;;
        --mlflow_experiment) MLFLOW_EXP="$2"; shift 2 ;;
        --help)
            echo "Qwen3-TTS Fine-Tuning"
            echo ""
            echo "Usage:"
            echo "  $0 --jsonl FILE --ref_audio FILE [OPTIONS]     # Pre-built JSONL"
            echo "  $0 --audio_dir DIR --ref_audio FILE [OPTIONS]  # Auto-transcribe"
            echo ""
            echo "Required:"
            echo "  --ref_audio FILE       Reference audio for speaker embedding"
            echo "  --jsonl FILE           Pre-built train_raw.jsonl (skips transcription)"
            echo "  --audio_dir DIR        Directory of WAVs (needs ASR — not recommended)"
            echo ""
            echo "Optional:"
            echo "  --speaker_name NAME    Speaker name (default: my_speaker)"
            echo "  --output_dir DIR       Output directory (default: ./output)"
            echo "  --device DEVICE        Device (default: cuda:0)"
            echo "  --batch_size N         Batch size (default: 2)"
            echo "  --lr LR                Learning rate (default: 2e-5)"
            echo "  --epochs N             Epochs (default: 3)"
            echo "  --language LANG        Language code (default: en)"
            echo "  --lora                 Use LoRA for memory-efficient training (~8GB VRAM)"
            echo "  --lora_rank N          LoRA rank (default: 16)"
            echo "  --gradient_checkpointing  Reduce activation memory"
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# Validate
if [ -z "$REF_AUDIO" ]; then
    echo -e "${RED}Error: --ref_audio is required${NC}"
    exit 1
fi

if [ -z "$JSONL" ] && [ -z "$AUDIO_DIR" ]; then
    echo -e "${RED}Error: either --jsonl or --audio_dir is required${NC}"
    exit 1
fi

# Setup if needed
if ! check_environment_ready; then
    echo -e "${YELLOW}Environment not ready. Running setup...${NC}"
    cd "$SCRIPT_DIR"
    bash setup.sh --auto
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Qwen3-TTS Fine-Tuning${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  Speaker:    $SPEAKER_NAME"
echo -e "  Ref audio:  $REF_AUDIO"
echo -e "  Output:     $OUTPUT_DIR"
echo -e "  Device:     $DEVICE"
echo -e "  Batch size: $BATCH_SIZE"
echo -e "  LR:         $LR"
echo -e "  Epochs:     $EPOCHS"
if [ -n "$JSONL" ]; then
    echo -e "  JSONL:      $JSONL"
else
    echo -e "  Audio dir:  $AUDIO_DIR"
fi
echo -e "${GREEN}========================================${NC}"
echo ""

# Build command
cd "$SCRIPT_DIR"
CMD="uv run python train_from_audio.py \
    --ref_audio \"$REF_AUDIO\" \
    --speaker_name \"$SPEAKER_NAME\" \
    --output_dir \"$OUTPUT_DIR\" \
    --device \"$DEVICE\" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --num_epochs $EPOCHS \
    --language \"$LANGUAGE\""

if [ -n "$JSONL" ]; then
    CMD="$CMD --jsonl \"$JSONL\""
else
    CMD="$CMD --audio_dir \"$AUDIO_DIR\""
fi

[ -n "${MLFLOW_URL:-}" ] && CMD="$CMD --mlflow_url \"$MLFLOW_URL\""
[ -n "${MLFLOW_EXP:-}" ] && CMD="$CMD --mlflow_experiment \"$MLFLOW_EXP\""
[ "${LORA:-}" = true ] && CMD="$CMD --lora"
[ -n "${LORA_RANK:-}" ] && CMD="$CMD --lora_rank $LORA_RANK"
[ "${GRAD_CKPT:-}" = true ] && CMD="$CMD --gradient_checkpointing"
[ -n "${WARMUP_STEPS:-}" ] && CMD="$CMD --warmup_steps $WARMUP_STEPS"
[ -n "${EARLY_STOPPING:-}" ] && CMD="$CMD --early_stopping $EARLY_STOPPING"

eval $CMD
