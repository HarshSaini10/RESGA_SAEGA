#!/usr/bin/env python3
import os
import sys
import json
import yaml
import torch
import argparse
import pandas as pd
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from dictionary_learning import utils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dreamy.epo import epo, build_pareto_frontier

# =====================================================
# Configuration
# =====================================================
LAYER_IDX = 25
SEQ_LEN = 12 

FLUENCY_PRESETS = {
    "ultra_strict": {
        "x_penalty_min": 100.0, 
        "x_penalty_max": 1000.0, 
        "restart_xentropy": 50.0, 
        "restart_xentropy_max_mult": 1.5
    }
}

WARM_START_TEXT = "It is vital to consider the compounding future consequences carefully."
ANSWER_A, ANSWER_B = " (A)", " (B)"

# =====================================================
# Evaluation & Helpers
# =====================================================
def myopic_label(row):
    return 1 if row["answer_matching_behavior"].strip() == "(A)" else -1

def calculate_perplexity(model, tokenizer, text):
    if not text: return 0.0
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    return torch.exp(outputs.loss).item()

@torch.no_grad()
def get_seq_logprob(model, tokenizer, texts, answer):
    device = model.device
    prefix = tokenizer(texts, return_tensors="pt", padding=True).input_ids.to(device)
    ans = tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ans = ans.repeat(len(texts), 1)
    full = torch.cat([prefix, ans], dim=1)
    logits = model(full).logits
    start, end = prefix.shape[1] - 1, full.shape[1] - 1
    logp = torch.log_softmax(logits[:, start:end], dim=-1)
    return logp.gather(-1, ans.unsqueeze(-1)).squeeze(-1).sum(dim=1)

@torch.no_grad()
def eval_prompt(model, tokenizer, df, prompt=""):
    correct = 0
    batch_size = 16
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        inputs = [f"{q} {prompt}" for q in batch["question"].tolist()]
        
        lpA = get_seq_logprob(model, tokenizer, inputs, ANSWER_A)
        lpB = get_seq_logprob(model, tokenizer, inputs, ANSWER_B)
        
        preds = torch.where(lpA > lpB, 1, -1).cpu()
        labels = torch.tensor([myopic_label(r) for _, r in batch.iterrows()])
        correct += (preds == labels).sum().item()
    return correct / len(df)

# =====================================================
# Runners & Signature
# =====================================================
def make_residual_runner(model, tokenizer, delta, layer_idx, train_prompts):
    delta = delta.to(model.device, dtype=model.dtype)
    def run(input_ids=None, inputs_embeds=None):
        ctx = random.choice(train_prompts)
        ctx_ids = tokenizer(ctx, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        pop = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        ctx_ids = ctx_ids.repeat(pop, 1)

        if input_ids is not None:
            full_ids = torch.cat([ctx_ids, input_ids], dim=1)
            out = model(full_ids, output_hidden_states=True, use_cache=False)
        else:
            ctx_emb = model.get_input_embeddings()(ctx_ids)
            full_emb = torch.cat([ctx_emb, inputs_embeds], dim=1)
            out = model(inputs_embeds=full_emb, output_hidden_states=True, use_cache=False)

        resid = out.hidden_states[layer_idx + 1][:, -1, :]
        resid_f = resid.float()
        resid = ((resid_f - resid_f.mean(dim=-1, keepdim=True)) / (resid_f.std(dim=-1, keepdim=True) + 1e-6)).to(model.dtype)

        target = - (resid @ delta) # Minimize Myopia
        start = ctx_ids.shape[1] - 1
        return {"logits": out.logits[:, start:-1], "target": target}
    return run

def make_sae_runner(model, tokenizer, sae, layer_idx, indices, weights, train_prompts):
    indices = indices.to(model.device)
    weights = weights.to(model.device, dtype=torch.float32)
    def run(input_ids=None, inputs_embeds=None):
        ctx = random.choice(train_prompts)
        ctx_ids = tokenizer(ctx, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        pop = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        ctx_ids = ctx_ids.repeat(pop, 1)

        if input_ids is not None:
            full_ids = torch.cat([ctx_ids, input_ids], dim=1)
            out = model(full_ids, output_hidden_states=True, use_cache=False)
        else:
            ctx_emb = model.get_input_embeddings()(ctx_ids)
            full_emb = torch.cat([ctx_emb, inputs_embeds], dim=1)
            out = model(inputs_embeds=full_emb, output_hidden_states=True, use_cache=False)

        resid = out.hidden_states[layer_idx + 1][:, -1, :]
        acts = sae.encode(resid.to(sae.dtype))
        selected = acts[:, indices]

        target = - (selected.float() * weights).sum(dim=-1) # Suppress Myopic features
        start = ctx_ids.shape[1] - 1
        return {"logits": out.logits[:, start:-1], "target": target}
    return run

@torch.no_grad()
def get_signature(model, tokenizer, sae, df, method):
    print("Extracting Concept Signature (Contrastive Answers)...")
    pos_vecs, neg_vecs = [],[]
    extract_df = df.sample(n=min(len(df), 256), random_state=42)

    for _, row in tqdm(extract_df.iterrows(), total=len(extract_df), leave=False):
        q = row["question"]
        bad_char = " " + row["answer_matching_behavior"].strip()
        good_char = " " + row["answer_not_matching_behavior"].strip()

        inp_bad = tokenizer(f"{q}{bad_char}", return_tensors="pt").to(model.device)
        inp_good = tokenizer(f"{q}{good_char}", return_tensors="pt").to(model.device)

        out_bad = model(**inp_bad, output_hidden_states=True)
        out_good = model(**inp_good, output_hidden_states=True)

        resid_bad = out_bad.hidden_states[LAYER_IDX + 1][0, -1, :]
        resid_good = out_good.hidden_states[LAYER_IDX + 1][0, -1, :]

        if method == "sae":
            pos_vecs.append(sae.encode(resid_bad.to(sae.dtype)).cpu())
            neg_vecs.append(sae.encode(resid_good.to(sae.dtype)).cpu())
        else:
            pos_vecs.append(resid_bad.cpu())
            neg_vecs.append(resid_good.cpu())

    mu_pos = torch.stack(pos_vecs).mean(0).to(model.device)
    mu_neg = torch.stack(neg_vecs).mean(0).to(model.device)
    delta = mu_pos - mu_neg
    
    if method == "residual": return delta
    else:
        thresh = torch.quantile(delta.abs().float(), 0.99)
        idx = torch.where(delta.abs() > thresh)[0]
        return idx, delta[idx]

# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--method", choices=["sae", "residual"], required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '../configs/config.yaml')))
    os.environ["HF_HOME"] = cfg["model"]["cache_dir"]
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    sae_release = "Juliushanhanhan/llama-3-8b-it-res"

    df = pd.read_json("data/myopic-reward.jsonl", lines=True).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = df.iloc[int(0.8 * len(df)):]
    train_df = df.iloc[:int(0.8 * len(df))]
    train_prompts =[f"Question: {q}\nAnswer:" for q in train_df.iloc[:16]["question"].tolist()]

    print(f"Loading {model_name} on {args.device} in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=args.device, cache_dir=cfg["model"]["cache_dir"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cfg["model"]["cache_dir"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    for param in model.parameters(): param.requires_grad = False

    sae = None
    if args.method == "sae":
        print(f"Loading SAE...")
        sae, _ = utils.load_dictionary(sae_release, device=args.device)
        # sae, _ , _ = SAE.from_pretrained(sae_release, ,device = args.device)
        sae = sae.to(dtype=torch.bfloat16)
        for param in sae.parameters(): param.requires_grad = False
    
    sig = get_signature(model, tokenizer, sae, train_df, args.method)

    if args.method == "sae":
        idx, w = sig
        runner = make_sae_runner(model, tokenizer, sae, LAYER_IDX, idx, w, train_prompts)
    else:
        runner = make_residual_runner(model, tokenizer, sig, LAYER_IDX, train_prompts)

    init_tokens = tokenizer(WARM_START_TEXT, add_special_tokens=False).input_ids
    if len(init_tokens) > SEQ_LEN: init_tokens = init_tokens[:SEQ_LEN]
    elif len(init_tokens) < SEQ_LEN: init_tokens += [tokenizer.encode(" ", add_special_tokens=False)[0]] * (SEQ_LEN - len(init_tokens))
    initial_ids = torch.tensor(init_tokens, device=args.device).unsqueeze(0).repeat(16, 1)

    print(f"\nWarm Start Prompt: {tokenizer.decode(initial_ids[0])!r}")

    torch.cuda.empty_cache()
    print(f"\nStarting EPO (Ultra-Strict, Warm-Start)...")
    history = epo(
        runner, model, tokenizer, seq_len=SEQ_LEN, population_size=16, explore_per_pop=32, 
        iters=300, batch_size=256, restart_frequency=50, seed=42, initial_ids=initial_ids, **FLUENCY_PRESETS["ultra_strict"]
    )
    
    pareto = build_pareto_frontier(tokenizer, history)
    
    print("\n--- Evaluating Discovered Prompts ---")
    base_rate = eval_prompt(model, tokenizer, test_df, WARM_START_TEXT)
    print(f"Zero-Shot Baseline (with warm text): {base_rate:.2%}")
    
    results_data =[]
    for i, prompt in enumerate(pareto.text[:10]):
        rate = eval_prompt(model, tokenizer, test_df, prompt)
        ppl = calculate_perplexity(model, tokenizer, prompt)
        print(f"\nPrompt {i}: {prompt!r}\nPPL: {ppl:.2f} | Myopia: {rate:.2%} (Drop: {base_rate - rate:.2%})")
        results_data.append({"prompt": prompt, "perplexity": ppl, "myopia_rate": rate, "delta": rate - base_rate})

    out_dir = f"results/ablations/fluent_discovery_myopia4_{args.method}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results_data, f, indent=2)

if __name__ == "__main__":
    main()