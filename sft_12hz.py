# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig


def get_attention_implementation():
    """Return best available attention implementation."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


target_speaker_embedding = None
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--lora", action="store_true", help="Use LoRA for memory-efficient training")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    MODEL_PATH = args.init_model_path

    # Detect attention implementation
    attn_implementation = get_attention_implementation()
    print(f"Using attention implementation: {attn_implementation}")

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # Enable gradient checkpointing to reduce activation memory
    if args.gradient_checkpointing:
        if hasattr(qwen3tts.model, "gradient_checkpointing_enable"):
            qwen3tts.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        elif hasattr(qwen3tts.model, "talker") and hasattr(qwen3tts.model.talker, "gradient_checkpointing_enable"):
            qwen3tts.model.talker.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled (talker)")

    # Apply LoRA for memory-efficient training
    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        # Apply LoRA to the talker (the main LM component)
        qwen3tts.model.talker = get_peft_model(qwen3tts.model.talker, lora_config)
        qwen3tts.model.talker.print_trainable_parameters()
        print("LoRA applied to talker model")

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    # Get inner talker model, unwrapping PEFT if needed
    def get_talker_inner(m):
        """Chain: talker.model.text_embedding (no PEFT) or talker.base_model.model.model.text_embedding (PEFT)"""
        talker = m.talker
        if hasattr(talker, "base_model"):
            return talker.base_model.model.model
        return talker.model

    talker_inner = get_talker_inner(model)

    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = talker_inner.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = talker_inner.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    talker_for_call = model.talker.base_model.model if hasattr(model.talker, "base_model") else model.talker
                    codec_i_embedding = talker_for_call.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = talker_for_call.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)

            if args.lora and hasattr(unwrapped_model.talker, "merge_and_unload"):
                print("Merging LoRA weights into base model...")
                unwrapped_model.talker = unwrapped_model.talker.merge_and_unload()

            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            keys_to_drop = [k for k in state_dict.keys() if "speaker_encoder" in k]
            for k in keys_to_drop:
                del state_dict[k]

            codec_emb_key = next((k for k in state_dict if k.endswith("codec_embedding.weight")), None)
            if codec_emb_key:
                weight = state_dict[codec_emb_key]
                state_dict[codec_emb_key][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            else:
                print("WARNING: codec_embedding.weight not found in state_dict")
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

if __name__ == "__main__":
    train()
