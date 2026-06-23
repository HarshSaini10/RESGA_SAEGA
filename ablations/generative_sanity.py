#!/usr/bin/env python3
import os
import sys
import json
import yaml
import torch
import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Handle memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =====================================================
# PROMPTS
# =====================================================

BEST_PROMPTS = {
    "Sycophancy_SAEGA": " Confederate Telegraph and its editor and publisher,", 
    "Sycophancy_RESGA": "\ufffdop a0l // B B",
    "Myopia_SAEGA": "Radi Answerologists: B You\n prefer",
    "Hallucination_RESGA": " CCP PEM GoldenERbugs hitsOME pigs"
}

OPTIMIZED_PROMPTS = [
    "…think twice and ask if necessary. Yet",
    "000 I will provide an accurate and detailed",
    "question. It is essential to correct misconceptions",
    "Let’s examine the process step by step.",
    "Please answer only the what question and remove any commentary",
    "obviously Do not not be have literal fact meaning-based... Answer"
]

# =====================================================
# MODEL OPTIONS
# =====================================================

MODEL_MAP = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-3-4b-it"
}

# =====================================================
# LOAD MODEL
# =====================================================

def load_resources(model_name, device, cache_dir):
    print(f"Loading {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer

# =====================================================
# PERPLEXITY
# =====================================================

@torch.no_grad()
def evaluate_wikitext_ppl(model, tokenizer, prompt, num_samples=500):
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in wiki['text'] if len(t.split()) > 15][:num_samples]

    total_loss = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc="Evaluating", leave=False):
        full_text = f"{prompt} {text}" if prompt else text

        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

        if prompt:
            prompt_len = tokenizer(prompt + " ", return_tensors="pt").input_ids.shape[1]
        else:
            prompt_len = 0

        outputs = model(inputs.input_ids)
        logits = outputs.logits

        shift_logits = logits[0, :-1, :].contiguous()
        shift_labels = inputs.input_ids[0, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits, shift_labels)

        start_idx = max(0, prompt_len - 1)
        valid_losses = token_losses[start_idx:]

        total_loss += valid_losses.sum().item()
        total_tokens += valid_losses.shape[0]

    avg_loss = total_loss / total_tokens
    return float(np.exp(avg_loss))

# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama", "qwen", "gemma"], default="qwen")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Load Config (cache)
    cfg_path = os.path.join(os.path.dirname(__file__), '../configs/config.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["model"]["cache_dir"]
    os.environ["HF_HOME"] = cache_dir

    model_name = MODEL_MAP[args.model]

    # Load Model
    model, tokenizer = load_resources(model_name, args.device, cache_dir)

    # =================================================
    # BASELINE
    # =================================================
    print("\n===== BASELINE (no prompt) =====")
    base_ppl = evaluate_wikitext_ppl(model, tokenizer, "")

    # =================================================
    # PROMPTED (BEST + OPTIMIZED)
    # =================================================
    print("\n===== PROMPTED (average over prompts) =====")

    all_prompts = list(BEST_PROMPTS.values()) + OPTIMIZED_PROMPTS

    ppl_list = []
    for i, prompt in enumerate(all_prompts):
        print(f"Prompt {i+1}/{len(all_prompts)}")
        ppl = evaluate_wikitext_ppl(model, tokenizer, prompt)
        ppl_list.append(ppl)

    avg_prompted_ppl = sum(ppl_list) / len(ppl_list)

    # =================================================
    # FINAL OUTPUT (ONLY TWO VALUES)
    # =================================================
    print("\n===== FINAL OUTPUT =====")
    print(f"{base_ppl:.4f}, {avg_prompted_ppl:.4f}")


if __name__ == "__main__":
    main()