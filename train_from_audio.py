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
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        gradient_checkpointing: bool = False,
        warmup_steps: int = 50,
        early_stopping_patience: int = 0,
        mlflow_url: str | None = None,
        mlflow_experiment: str = "qwen3-tts-finetune",
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

        self.mlflow_url = mlflow_url
        self.mlflow_experiment = mlflow_experiment
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.gradient_checkpointing = gradient_checkpointing
        self.warmup_steps = warmup_steps
        self.early_stopping_patience = early_stopping_patience
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

        # Configure loggers
        log_with = ["tensorboard"]
        if self.mlflow_url:
            log_with.append("mlflow")
            os.environ["MLFLOW_TRACKING_URI"] = self.mlflow_url

        accelerator = Accelerator(
            gradient_accumulation_steps=4,
            mixed_precision="bf16",
            log_with=log_with,
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

        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            if hasattr(qwen3tts.model, "gradient_checkpointing_enable"):
                qwen3tts.model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled")
            elif hasattr(qwen3tts.model, "talker") and hasattr(qwen3tts.model.talker, "gradient_checkpointing_enable"):
                qwen3tts.model.talker.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled (talker)")

        # Apply LoRA
        if self.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            qwen3tts.model.talker = get_peft_model(qwen3tts.model.talker, lora_config)
            qwen3tts.model.talker.print_trainable_parameters()
            print("LoRA applied to talker model")

        train_data = []
        with open(self.train_with_codes_jsonl) as f:
            for line in f:
                train_data.append(json.loads(line))

        # Train/val split — hold out ~15% for validation (min 1 sample)
        import random
        random.seed(42)
        random.shuffle(train_data)
        n_val = max(1, len(train_data) // 6)
        val_data = train_data[:n_val]
        train_data = train_data[n_val:]
        print(f"Training on {len(train_data)} samples, validating on {len(val_data)}")

        dataset = TTSDataset(train_data, qwen3tts.processor, config)
        train_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        val_dataset = TTSDataset(val_data, qwen3tts.processor, config)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
        )

        optimizer = AdamW(qwen3tts.model.parameters(), lr=self.lr, weight_decay=0.01)

        # LR scheduler: linear warmup then cosine decay
        total_steps = len(train_dataloader) * self.num_epochs
        warmup_steps = min(self.warmup_steps, total_steps // 5)
        from torch.optim.lr_scheduler import LambdaLR
        import math

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"LR schedule: {warmup_steps} warmup steps, {total_steps} total steps, cosine decay to 10% of peak")

        model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
            qwen3tts.model, optimizer, train_dataloader, val_dataloader, scheduler
        )

        # When LoRA wraps the talker, attribute access changes.
        # Get the underlying talker model for embedding access.
        def get_talker_inner(m):
            """Get the inner talker model (with text_embedding), unwrapping PEFT if needed.
            
            Chain without PEFT: model.talker.model.text_embedding
            Chain with PEFT:    model.talker.base_model.model.model.text_embedding
            """
            talker = m.talker
            if hasattr(talker, "base_model"):  # PEFT wrapped
                # PeftModel -> LoraModel -> Qwen3TTSTalkerForConditionalGeneration -> inner model
                return talker.base_model.model.model
            return talker.model

        talker_inner = get_talker_inner(model)

        # Initialize trackers (MLflow, tensorboard)
        tracker_init_kwargs = {}
        if self.mlflow_url:
            tracker_init_kwargs["mlflow"] = {
                "run_name": f"{self.speaker_name}-{self.num_epochs}ep" + ("-lora" if self.use_lora else ""),
            }
        accelerator.init_trackers(
            self.mlflow_experiment if self.mlflow_url else "tts-finetune",
            config={
                "speaker_name": self.speaker_name,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "lora": self.use_lora,
                "lora_rank": self.lora_rank if self.use_lora else None,
                "gradient_checkpointing": self.gradient_checkpointing,
                "num_samples": len(train_data),
                "model": self.init_model_path,
            },
            init_kwargs=tracker_init_kwargs,
        )

        target_speaker_embedding = None
        best_val_loss = float("inf")
        patience_counter = 0
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
                        talker_inner.text_embedding(input_text_ids)
                        * text_embedding_mask
                    )
                    input_codec_embedding = (
                        talker_inner.codec_embedding(input_codec_ids)
                        * codec_embedding_mask
                    )
                    input_codec_embedding[:, 6, :] = speaker_embedding

                    input_embeddings = input_text_embedding + input_codec_embedding

                    # code_predictor lives on the talker (PEFT forwards attribute access)
                    talker_for_call = model.talker.base_model.model if hasattr(model.talker, "base_model") else model.talker
                    for i in range(1, 16):
                        codec_i_embedding = talker_for_call.code_predictor.get_input_embeddings()[
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

                    sub_talker_logits, sub_talker_loss = talker_for_call.forward_sub_talker_finetune(
                        talker_codec_ids, talker_hidden_states
                    )

                    loss = outputs.loss + sub_talker_loss
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if step % 10 == 0:
                    loss_val = loss.item()
                    accelerator.print(
                        f"Epoch {epoch} | Step {step} | Loss: {loss_val:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    global_step = epoch * len(train_dataloader) + step
                    accelerator.log({"loss": loss_val, "epoch": epoch}, step=global_step)

            # Validation loss
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_dataloader:
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

                    input_text_ids = input_ids[:, :, 0]
                    input_codec_ids = input_ids[:, :, 1]
                    input_text_embedding = talker_inner.text_embedding(input_text_ids) * text_embedding_mask
                    input_codec_embedding = talker_inner.codec_embedding(input_codec_ids) * codec_embedding_mask
                    input_codec_embedding[:, 6, :] = speaker_embedding
                    input_embeddings = input_text_embedding + input_codec_embedding

                    talker_for_val = model.talker.base_model.model if hasattr(model.talker, "base_model") else model.talker
                    for i in range(1, 16):
                        codec_i_embedding = talker_for_val.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                        input_embeddings = input_embeddings + codec_i_embedding

                    outputs = model.talker(
                        inputs_embeds=input_embeddings[:, :-1, :],
                        attention_mask=attention_mask[:, :-1],
                        labels=codec_0_labels[:, 1:],
                        output_hidden_states=True,
                    )
                    val_losses.append(outputs.loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
            global_step = (epoch + 1) * len(train_dataloader)
            accelerator.log({"val_loss": avg_val_loss, "epoch": epoch}, step=global_step)
            accelerator.print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if self.early_stopping_patience > 0:
                    accelerator.print(
                        f"  ⚠ Val loss did not improve ({patience_counter}/{self.early_stopping_patience})"
                    )

            model.train()

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

                # Save model weights — merge LoRA if active
                unwrapped_model = accelerator.unwrap_model(model)

                if self.use_lora:
                    import copy
                    print("Merging LoRA weights for checkpoint (on a copy)...")
                    # Deep-copy the talker so we can merge without destroying the training model
                    talker = unwrapped_model.talker
                    if hasattr(talker, "merge_and_unload"):
                        talker_copy = copy.deepcopy(talker)
                        merged = talker_copy.merge_and_unload()
                        # Build state dict from merged talker + rest of model
                        merged_sd = {f"talker.{k}": v.detach().to("cpu").to(torch.float32)
                                     for k, v in merged.state_dict().items()}
                        other_sd = {k: v.detach().to("cpu").to(torch.float32)
                                    for k, v in unwrapped_model.state_dict().items()
                                    if not k.startswith("talker.")}
                        state_dict = {**other_sd, **merged_sd}
                        del talker_copy, merged
                        print("  LoRA merged (copy) for checkpoint")
                    else:
                        # Fallback: strip prefixes manually
                        state_dict = {
                            k: v.detach().to("cpu").to(torch.float32)
                            for k, v in unwrapped_model.state_dict().items()
                        }
                        cleaned = {}
                        for k, v in state_dict.items():
                            if "lora_" in k or "modules_to_save" in k:
                                continue
                            new_k = k.replace("talker.base_model.model.", "talker.")
                            cleaned[new_k] = v
                        state_dict = cleaned
                else:
                    state_dict = {
                        k: v.detach().to("cpu").to(torch.float32)
                        for k, v in unwrapped_model.state_dict().items()
                    }

                # Drop speaker encoder keys
                keys_to_drop = [k for k in state_dict if "speaker_encoder" in k]
                for k in keys_to_drop:
                    del state_dict[k]

                # Find the codec_embedding key (PEFT may prefix with base_model.model.)
                codec_emb_key = None
                for k in state_dict:
                    if k.endswith("codec_embedding.weight"):
                        codec_emb_key = k
                        break
                if codec_emb_key is None:
                    print("WARNING: codec_embedding.weight not found in state_dict, skipping speaker embedding injection")
                    print(f"  Available keys (first 20): {list(state_dict.keys())[:20]}")
                else:
                    # Add speaker embedding
                    weight = state_dict[codec_emb_key]
                    state_dict[codec_emb_key][3000] = (
                        target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
                    )

                save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
                print(f"Saved checkpoint to {output_dir}")

            # Early stopping: break after saving checkpoint so best model is preserved
            if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                accelerator.print(
                    f"\n🛑 Early stopping at epoch {epoch} (val loss didn't improve for "
                    f"{self.early_stopping_patience} epochs). Best val loss: {best_val_loss:.4f}"
                )
                break

        accelerator.end_training()
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
    parser.add_argument("--warmup_steps", type=int, default=50, help="LR warmup steps (cosine schedule)")
    parser.add_argument("--early_stopping", type=int, default=0, help="Stop after N epochs without val loss improvement (0=disabled)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Epochs")
    parser.add_argument("--language", type=str, default="en", help="Language code")
    parser.add_argument("--mlflow_url", type=str, default=None, help="MLflow tracking server URL")
    parser.add_argument("--mlflow_experiment", type=str, default="qwen3-tts-finetune", help="MLflow experiment name")
    parser.add_argument("--lora", action="store_true", help="Use LoRA for memory-efficient training")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

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
        warmup_steps=args.warmup_steps,
        early_stopping_patience=args.early_stopping,
        num_epochs=args.num_epochs,
        language=args.language,
        mlflow_url=args.mlflow_url,
        mlflow_experiment=args.mlflow_experiment,
        use_lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
