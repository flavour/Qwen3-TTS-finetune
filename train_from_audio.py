#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-TTS Fine-Tuning Script

Supports two modes:
  1. --jsonl: Use pre-built train_raw.jsonl (our workflow, hand-corrected transcripts)
  2. --audio_dir: Auto-discover WAV files (transcripts must already be in JSONL)

Pipeline:
  1. Load/create train_raw.jsonl
  2. Extract audio_codes using Qwen3-TTS Tokenizer
  3. Fine-tune the model

Based on https://github.com/sruckh/Qwen3-TTS-finetune
Modified: removed WhisperX dependency, added --jsonl input, uses uv.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import torch
import torchaudio
from tqdm import tqdm


def configure_hf_cache():
    """Configure HuggingFace cache."""
    script_dir = Path(__file__).parent.absolute()
    hf_cache = script_dir / ".venv" / "hf_cache"
    if hf_cache.parent.exists():
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_cache))


def get_attention_implementation():
    """Return best available attention implementation."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


configure_hf_cache()


class Qwen3TTSPipeline:
    """End-to-end pipeline for Qwen3-TTS fine-tuning."""

    def __init__(
        self,
        ref_audio: str,
        speaker_name: str,
        audio_dir: str | None = None,
        jsonl: str | None = None,
        output_dir: str = "./output",
        device: str = "cuda:0",
        tokenizer_model_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        init_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        batch_size: int = 2,
        lr: float = 2e-5,
        num_epochs: int = 3,
        language: str = "en",
    ):
        self.ref_audio = Path(ref_audio)
        self.speaker_name = speaker_name
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.jsonl = Path(jsonl) if jsonl else None
        self.output_dir = Path(output_dir)
        self.device = device
        self.tokenizer_model_path = tokenizer_model_path
        self.init_model_path = init_model_path
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.language = language

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_raw_jsonl = self.output_dir / "train_raw.jsonl"
        self.train_with_codes_jsonl = self.output_dir / "train_with_codes.jsonl"
        self.attn_implementation = get_attention_implementation()

    def validate_audio_files(self) -> List[Path]:
        """Find and validate all WAV files in the audio directory."""
        if not self.audio_dir or not self.audio_dir.exists():
            raise ValueError(f"Audio directory not found: {self.audio_dir}")

        wav_files = list(self.audio_dir.glob("*.wav")) + list(self.audio_dir.glob("*.WAV"))
        if not wav_files:
            raise ValueError(f"No WAV files found in {self.audio_dir}")

        if not self.ref_audio.exists():
            raise ValueError(f"Reference audio not found: {self.ref_audio}")

        valid_files = []
        for wav_path in tqdm(wav_files, desc="Validating audio files"):
            try:
                torchaudio.load(str(wav_path))
                valid_files.append(wav_path)
            except Exception as e:
                print(f"Warning: Could not load {wav_path}: {e}")

        print(f"Found {len(valid_files)} valid audio files")
        return valid_files

    def load_or_create_jsonl(self) -> None:
        """Load pre-built JSONL or create from audio directory."""
        print(f"\n{'='*60}")
        print("STEP 1: Preparing train_raw.jsonl")
        print(f"{'='*60}\n")

        if self.jsonl and self.jsonl.exists():
            # Use pre-built JSONL — just copy/link to output dir
            print(f"Using pre-built JSONL: {self.jsonl}")

            # Read and inject ref_audio into each entry
            entries = []
            with open(self.jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    # Ensure ref_audio is set
                    if "ref_audio" not in entry:
                        entry["ref_audio"] = str(self.ref_audio.resolve())
                    entries.append(entry)

            # Resolve relative audio paths against JSONL's parent dir
            jsonl_dir = self.jsonl.parent.resolve()
            for entry in entries:
                audio_path = Path(entry["audio"])
                if not audio_path.is_absolute():
                    resolved = jsonl_dir / audio_path
                    entry["audio"] = str(resolved.resolve())

            with open(self.train_raw_jsonl, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"Loaded {len(entries)} entries → {self.train_raw_jsonl}")

        elif self.audio_dir:
            # Create JSONL from audio files (no transcription — user must provide)
            audio_files = self.validate_audio_files()
            print(f"\nWarning: Creating JSONL from audio files WITHOUT transcription.")
            print(f"For best results, use --jsonl with hand-corrected transcripts.\n")

            # This is a fallback — audio files need accompanying text somehow
            raise ValueError(
                "Audio-only mode requires --jsonl with transcripts. "
                "Use prep_training.py to create train_raw.jsonl from manifest.csv first."
            )
        else:
            raise ValueError("Either --jsonl or --audio_dir is required")

    def prepare_data(self) -> None:
        """Extract audio_codes using Qwen3-TTS Tokenizer."""
        print(f"\n{'='*60}")
        print("STEP 2: Preparing data (extracting audio_codes)")
        print(f"{'='*60}\n")

        from qwen_tts import Qwen3TTSTokenizer

        print(f"Loading tokenizer: {self.tokenizer_model_path}")
        tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
            self.tokenizer_model_path,
            device_map=self.device,
        )

        total_lines = []
        with open(self.train_raw_jsonl) as f:
            for line in f:
                total_lines.append(json.loads(line.strip()))

        final_lines = []
        batch_lines = []
        batch_audios = []
        BATCH_INFER_NUM = 32

        print(f"Processing {len(total_lines)} audio files...")

        for line in tqdm(total_lines, desc="Encoding audio"):
            batch_lines.append(line)
            batch_audios.append(line["audio"])

            if len(batch_lines) >= BATCH_INFER_NUM:
                enc_res = tokenizer_12hz.encode(batch_audios)
                for code, bl in zip(enc_res.audio_codes, batch_lines):
                    bl["audio_codes"] = code.cpu().tolist()
                    final_lines.append(bl)
                batch_lines.clear()
                batch_audios.clear()

        if batch_audios:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, bl in zip(enc_res.audio_codes, batch_lines):
                bl["audio_codes"] = code.cpu().tolist()
                final_lines.append(bl)

        with open(self.train_with_codes_jsonl, "w", encoding="utf-8") as f:
            for fl in final_lines:
                f.write(json.dumps(fl, ensure_ascii=False) + "\n")

        print(f"Created {self.train_with_codes_jsonl}")

        del tokenizer_12hz
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_model(self) -> None:
        """Run fine-tuning."""
        print(f"\n{'='*60}")
        print("STEP 3: Fine-tuning model")
        print(f"{'='*60}\n")

        from dataset import TTSDataset
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        from transformers import AutoConfig
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from accelerate import Accelerator
        from safetensors.torch import save_file

        logging_dir = self.output_dir / "logs"
        logging_dir.mkdir(parents=True, exist_ok=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=4,
            mixed_precision="bf16",
            log_with="tensorboard",
            project_dir=str(logging_dir),
        )

        print(f"Loading model: {self.init_model_path}")
        print(f"Attention: {self.attn_implementation}")
        qwen3tts = Qwen3TTSModel.from_pretrained(
            self.init_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.attn_implementation,
        )

        config = AutoConfig.from_pretrained(self.init_model_path)

        train_data = []
        with open(self.train_with_codes_jsonl) as f:
            for line in f:
                train_data.append(json.loads(line))

        print(f"Training on {len(train_data)} samples")

        dataset = TTSDataset(train_data, qwen3tts.processor, config)
        train_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        optimizer = AdamW(qwen3tts.model.parameters(), lr=self.lr, weight_decay=0.01)

        model, optimizer, train_dataloader = accelerator.prepare(
            qwen3tts.model, optimizer, train_dataloader
        )

        target_speaker_embedding = None
        model.train()

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    input_ids = batch["input_ids"]
                    codec_ids = batch["codec_ids"]
                    ref_mels = batch["ref_mels"]
                    text_embedding_mask = batch["text_embedding_mask"]
                    codec_embedding_mask = batch["codec_embedding_mask"]
                    attention_mask = batch["attention_mask"]
                    codec_0_labels = batch["codec_0_labels"]
                    codec_mask = batch["codec_mask"]

                    speaker_embedding = model.speaker_encoder(
                        ref_mels.to(model.device).to(model.dtype)
                    ).detach()

                    if target_speaker_embedding is None:
                        target_speaker_embedding = speaker_embedding

                    input_text_ids = input_ids[:, :, 0]
                    input_codec_ids = input_ids[:, :, 1]

                    input_text_embedding = (
                        model.talker.model.text_embedding(input_text_ids)
                        * text_embedding_mask
                    )
                    input_codec_embedding = (
                        model.talker.model.codec_embedding(input_codec_ids)
                        * codec_embedding_mask
                    )
                    input_codec_embedding[:, 6, :] = speaker_embedding

                    input_embeddings = input_text_embedding + input_codec_embedding

                    for i in range(1, 16):
                        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[
                            i - 1
                        ](codec_ids[:, :, i])
                        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                        input_embeddings = input_embeddings + codec_i_embedding

                    outputs = model.talker(
                        inputs_embeds=input_embeddings[:, :-1, :],
                        attention_mask=attention_mask[:, :-1],
                        labels=codec_0_labels[:, 1:],
                        output_hidden_states=True,
                    )

                    hidden_states = outputs.hidden_states[0][-1]
                    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                    talker_codec_ids = codec_ids[codec_mask]

                    sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                        talker_codec_ids, talker_hidden_states
                    )

                    loss = outputs.loss + sub_talker_loss
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                if step % 10 == 0:
                    accelerator.print(
                        f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                    )

            # Save checkpoint
            if accelerator.is_main_process:
                output_dir = os.path.join(str(self.output_dir), f"checkpoint-epoch-{epoch}")

                from huggingface_hub import snapshot_download
                if os.path.isdir(self.init_model_path):
                    model_cache_path = self.init_model_path
                else:
                    model_cache_path = snapshot_download(self.init_model_path)

                shutil.copytree(model_cache_path, output_dir, dirs_exist_ok=True)

                # Update config for custom voice
                config_file = os.path.join(output_dir, "config.json")
                with open(config_file, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)

                config_dict["tts_model_type"] = "custom_voice"
                talker_config = config_dict.get("talker_config", {})
                talker_config["spk_id"] = {self.speaker_name: 3000}
                talker_config["spk_is_dialect"] = {self.speaker_name: False}
                config_dict["talker_config"] = talker_config

                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

                # Save model weights
                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = {
                    k: v.detach().to("cpu").to(torch.float32)
                    for k, v in unwrapped_model.state_dict().items()
                }

                # Drop speaker encoder keys
                keys_to_drop = [k for k in state_dict if k.startswith("speaker_encoder")]
                for k in keys_to_drop:
                    del state_dict[k]

                # Add speaker embedding
                weight = state_dict["talker.model.codec_embedding.weight"]
                state_dict["talker.model.codec_embedding.weight"][3000] = (
                    target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
                )

                save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
                print(f"Saved checkpoint to {output_dir}")

        print("\nTraining complete!")

    def run(self) -> None:
        """Run the complete pipeline."""
        print(f"\n{'='*60}")
        print("Qwen3-TTS Fine-Tuning Pipeline")
        print(f"{'='*60}\n")

        self.load_or_create_jsonl()
        self.prepare_data()
        self.train_model()

        print(f"\n{'='*60}")
        print("Pipeline complete!")
        print(f"Checkpoints saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Fine-Tuning Pipeline")

    parser.add_argument("--audio_dir", type=str, help="Directory containing WAV files")
    parser.add_argument("--jsonl", type=str, help="Pre-built train_raw.jsonl (preferred)")
    parser.add_argument("--ref_audio", type=str, required=True, help="Reference audio (WAV)")
    parser.add_argument("--speaker_name", type=str, default="my_speaker", help="Speaker name")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Epochs")
    parser.add_argument("--language", type=str, default="en", help="Language code")

    args = parser.parse_args()

    if not args.jsonl and not args.audio_dir:
        parser.error("Either --jsonl or --audio_dir is required")

    pipeline = Qwen3TTSPipeline(
        ref_audio=args.ref_audio,
        speaker_name=args.speaker_name,
        audio_dir=args.audio_dir,
        jsonl=args.jsonl,
        output_dir=args.output_dir,
        device=args.device,
        tokenizer_model_path=args.tokenizer_model_path,
        init_model_path=args.init_model_path,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        language=args.language,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
