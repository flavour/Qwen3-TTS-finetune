#!/usr/bin/env bash
# Qwen3-TTS One-Command Fine-Tuning Script
#
# This script provides the complete end-to-end pipeline:
# 1. Automatically sets up environment if needed
# 2. Transcribes audio files using WhisperX
# 3. Creates train_raw.jsonl
# 4. Prepares data (extracts audio_codes)
# 5. Fine-tunes the model
#
# Usage:
#   ./train.sh --audio_dir ./audio --ref_audio ./ref.wav --speaker_name my_voice

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if environment is ready
check_environment_ready() {
    # Check if venv exists
    if [ ! -d "$VENV_DIR" ]; then
        return 1
    fi

    # Check if key dependencies are installed
    if ! "$VENV_DIR/bin/python" -c "import torch; import whisperx; import qwen_tts" 2>/dev/null; then
        return 1
    fi

    return 0
}

# Default values
AUDIO_DIR=""
REF_AUDIO=""
SPEAKER_NAME="my_speaker"
OUTPUT_DIR="./output"
DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
BATCH_SIZE=2
LR=2e-5
EPOCHS=3
WHISPER_MODEL="large-v3"
LANGUAGE="auto"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --audio_dir)
            AUDIO_DIR="$2"
            shift 2
            ;;
        --ref_audio)
            REF_AUDIO="$2"
            shift 2
            ;;
        --speaker_name)
            SPEAKER_NAME="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --whisper_model)
            WHISPER_MODEL="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --help)
            echo "Qwen3-TTS One-Command Fine-Tuning"
            echo ""
            echo "Usage:"
            echo "  $0 --audio_dir DIR --ref_audio FILE [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --audio_dir DIR        Directory containing WAV files for training"
            echo "  --ref_audio FILE       Path to reference audio file (WAV)"
            echo ""
            echo "Optional:"
            echo "  --speaker_name NAME    Name for the speaker (default: my_speaker)"
            echo "  --output_dir DIR       Output directory (default: ./output)"
            echo "  --device DEVICE        Device to use (default: cuda:0)"
            echo "  --batch_size N         Batch size (default: 2)"
            echo "  --lr LR                Learning rate (default: 2e-5)"
            echo "  --epochs N             Number of epochs (default: 3)"
            echo "  --whisper_model MODEL  Whisper model size (default: large-v3)"
            echo "  --language LANG        Language code (default: auto)"
            echo ""
            echo "Example:"
            echo "  $0 --audio_dir ./my_recordings --ref_audio ./ref.wav --speaker_name alice"
            echo ""
            echo "The script will automatically set up the environment if needed."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$AUDIO_DIR" ] || [ -z "$REF_AUDIO" ]; then
    echo -e "${RED}Error: --audio_dir and --ref_audio are required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Check and setup environment automatically
if ! check_environment_ready; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Environment not ready. Running setup...${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    # Run setup in auto mode (non-interactive)
    bash "$SCRIPT_DIR/setup.sh" --auto

    echo ""
    echo -e "${GREEN}Setup complete. Continuing with training...${NC}"
    echo ""
fi

# Configure HuggingFace cache before activating venv
export HF_HOME="$VENV_DIR/hf_cache"

# Create cache directory if it doesn't exist
mkdir -p "$HF_HOME"

# Activate venv
source "$VENV_DIR/bin/activate"

# Print configuration
echo "=========================================="
echo "Qwen3-TTS End-to-End Fine-Tuning"
echo "=========================================="
echo "Audio directory:     $AUDIO_DIR"
echo "Reference audio:     $REF_AUDIO"
echo "Speaker name:        $SPEAKER_NAME"
echo "Output directory:    $OUTPUT_DIR"
echo "Device:              $DEVICE"
echo "Batch size:          $BATCH_SIZE"
echo "Learning rate:       $LR"
echo "Epochs:              $EPOCHS"
echo "Whisper model:       $WHISPER_MODEL"
echo "Language:            $LANGUAGE"
echo "HF cache:            $HF_HOME"
echo "=========================================="
echo ""

# Run the Python script
python "$SCRIPT_DIR/train_from_audio.py" \
    --audio_dir "$AUDIO_DIR" \
    --ref_audio "$REF_AUDIO" \
    --speaker_name "$SPEAKER_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --tokenizer_model_path "$TOKENIZER_MODEL_PATH" \
    --init_model_path "$INIT_MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_epochs "$EPOCHS" \
    --whisper_model "$WHISPER_MODEL" \
    --language "$LANGUAGE"

echo ""
echo "=========================================="
echo "Fine-tuning complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=========================================="
