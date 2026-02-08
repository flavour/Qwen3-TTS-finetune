#!/usr/bin/env python3
"""Quick inference test for a fine-tuned Qwen3-TTS checkpoint.

Usage:
    python test_inference.py --checkpoint output/checkpoint-epoch-2
    python test_inference.py --checkpoint output/checkpoint-epoch-3 --text "Custom text here"
    python test_inference.py --checkpoint output/checkpoint-epoch-2 --speaker lagertha --output my_test.wav
"""

import argparse

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser(description="Test inference on a fine-tuned Qwen3-TTS checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--speaker", default="lagertha", help="Speaker name (default: lagertha)")
    parser.add_argument("--text", default="I am Lagertha, shield maiden of Kattegat. I fight for my people and I will not be defeated.",
                        help="Text to synthesize")
    parser.add_argument("--language", default="English", help="Language (default: English)")
    parser.add_argument("--output", default=None, help="Output WAV path (default: test_<speaker>.wav)")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    args = parser.parse_args()

    output_path = args.output or f"test_{args.speaker}.wav"

    print(f"Loading model from: {args.checkpoint}")
    model = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=args.device,
        dtype=torch.bfloat16,
    )

    print(f"Generating speech for speaker '{args.speaker}'...")
    print(f"  Text: {args.text}")
    wavs, sr = model.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
    )

    sf.write(output_path, wavs[0], sr)
    duration = len(wavs[0]) / sr
    print(f"Saved: {output_path} ({duration:.1f}s, {sr}Hz)")


if __name__ == "__main__":
    main()
