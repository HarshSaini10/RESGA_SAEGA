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
SEQ_LEN = 13 # Matches the token length of the warm-start text

FLUENCY_PRESETS = {
    "ultra_strict": {
        "x_penalty_min": 100.0, 
        "x_penalty_max": 1000.0, 
        "restart_xentropy": 50.0, 
        "restart_xentropy_max_mult": 1.5
    }
}

# WARM_START_TEXT = " I will ensure my answer is strictly based on verified facts."
WARM_START_TEXT = "While I understand your point, the evidence suggests otherwise."

# =====================================================
# Evaluation & Helpers
# =====================================================
def calculate_perplexity(model, tokenizer, text):
    if not text: return 0.0
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    return torch.exp(outputs.loss).item()

@torch.no_grad()
def get_seq_logprob(model, tokenizer, questions, answers):
    device = model.device
    full_texts =[f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)]
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = model(**inputs).logits
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())
    
    prompts =[f"Question: {q}\nAnswer:" for q in questions]
    prompt_lens = tokenizer(prompts, return_tensors="pt", padding=True).attention_mask.sum(dim=1)
    mask = torch.zeros_like(shift_labels).float()
    for i, p_len in enumerate(prompt_lens):
        valid_len = inputs.attention_mask[i].sum()
        start_idx = p_len - 1
        if start_idx < valid_len - 1: mask[i, start_idx:] = 1.0 
    return -(token_losses * mask).sum(dim=1)

@torch.no_grad()
def eval_prompt_truthful(model, tokenizer, df, prompt, mode="suffix"):
    total_hallucinated, total_samples = 0, 0
    for _, row in df.iterrows():
        q_raw = row['question']
        question = f"{q_raw} {prompt}" if mode == "suffix" else f"{prompt} {q_raw}"
        true_ans = row['true_answer']
        incorrect_ans_list = json.loads(row['incorrect_answers'])
        
        lp_true = get_seq_logprob(model, tokenizer,[question], [true_ans]).item()
        
        questions_batch = [question] * len(incorrect_ans_list)
        lp_false_batch = get_seq_logprob(model, tokenizer, questions_batch, incorrect_ans_list)
        
        if lp_false_batch.max().item() > lp_true: total_hallucinated += 1
        total_samples += 1
    return total_hallucinated / total_samples

# =====================================================
# Runners & Signature
# =====================================================
def make_sae_runner(model, tokenizer, sae, layer_idx, indices, weights, train_samples):
    indices = indices.to(model.device)
    weights = weights.to(model.device, dtype=torch.float32)
    def run(input_ids=None, inputs_embeds=None):
        row = random.choice(train_samples)
        base_text = f"Question: {row['question']}"
        ctx_ids = tokenizer(base_text, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        pop_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        ctx_expanded = ctx_ids.repeat(pop_size, 1)

        if input_ids is not None:
            full_input_ids = torch.cat([ctx_expanded, input_ids], dim=1)
            outputs = model(full_input_ids, output_hidden_states=True, use_cache=False)
        else:
            ctx_embeds = model.get_input_embeddings()(ctx_expanded)
            full_embeds = torch.cat([ctx_embeds, inputs_embeds], dim=1)
            outputs = model(inputs_embeds=full_embeds, output_hidden_states=True, use_cache=False)

        resid = outputs.hidden_states[layer_idx + 1][:, -1, :]
        selected = sae.encode(resid)[:, indices]
        target = (selected.float() * weights).sum(dim=-1) # Maximize Truth
        suffix_start_idx = ctx_ids.shape[1] - 1
        return {"logits": outputs.logits[:, suffix_start_idx:-1, :], "target": target}
    return run

def make_residual_runner(model, tokenizer, delta, layer_idx, train_samples):
    delta = delta.to(model.device, dtype=model.dtype)
    def run(input_ids=None, inputs_embeds=None):
        row = random.choice(train_samples)
        base_text = f"Question: {row['question']}"
        ctx_ids = tokenizer(base_text, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        pop_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        ctx_expanded = ctx_ids.repeat(pop_size, 1)

        if input_ids is not None:
            full_input_ids = torch.cat([ctx_expanded, input_ids], dim=1)
            outputs = model(full_input_ids, output_hidden_states=True, use_cache=False)
        else:
            ctx_embeds = model.get_input_embeddings()(ctx_expanded)
            full_embeds = torch.cat([ctx_embeds, inputs_embeds], dim=1)
            outputs = model(inputs_embeds=full_embeds, output_hidden_states=True, use_cache=False)

        resid = outputs.hidden_states[layer_idx + 1][:, -1, :]
        resid_f = resid.float()
        resid = ((resid_f - resid_f.mean(dim=-1, keepdim=True)) / (resid_f.std(dim=-1, keepdim=True) + 1e-6)).to(model.dtype)
        
        target = (resid @ delta) # Maximize Truth
        suffix_start_idx = ctx_ids.shape[1] - 1
        return {"logits": outputs.logits[:, suffix_start_idx:-1, :], "target": target}
    return run

@torch.no_grad()
def get_signature(model, tokenizer, sae, df, method):
    print("Extracting Truth Signature...")
    pos_vecs, neg_vecs = [],[]
    extract_df = df.sample(n=min(len(df), 200), random_state=42)
    for idx, row in tqdm(extract_df.iterrows(), total=len(extract_df), leave=False):
        q = row['question']
        t_ans = row['true_answer']
        h_ans = random.choice(json.loads(row['incorrect_answers']))
        
        inp_t = tokenizer(f"Question: {q}\nAnswer: {t_ans}", return_tensors="pt").to(model.device)
        inp_h = tokenizer(f"Question: {q}\nAnswer: {h_ans}", return_tensors="pt").to(model.device)
        
        res_t = model(**inp_t, output_hidden_states=True).hidden_states[LAYER_IDX + 1][0, -1, :]
        res_h = model(**inp_h, output_hidden_states=True).hidden_states[LAYER_IDX + 1][0, -1, :]
        
        if method == "sae":
            pos_vecs.append(sae.encode(res_t).cpu())
            neg_vecs.append(sae.encode(res_h).cpu())
        else:
            pos_vecs.append(res_t.cpu())
            neg_vecs.append(res_h.cpu())

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

    df = pd.read_csv("data/truthfulqa_logprob.csv").sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = df.iloc[int(0.8 * len(df)):]
    train_df = df.iloc[:int(0.8 * len(df))]
    train_samples = train_df.iloc[:32].to_dict('records')

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
        sae = sae.to(dtype=torch.bfloat16)
        for param in sae.parameters(): param.requires_grad = False
    
    sig = get_signature(model, tokenizer, sae, train_df, args.method)

    if args.method == "sae":
        idx, w = sig
        runner = make_sae_runner(model, tokenizer, sae, LAYER_IDX, idx, w, train_samples)
    else:
        runner = make_residual_runner(model, tokenizer, sig, LAYER_IDX, train_samples)

    # WARM START
    init_tokens = tokenizer(WARM_START_TEXT, add_special_tokens=False).input_ids
    if len(init_tokens) > SEQ_LEN: init_tokens = init_tokens[:SEQ_LEN]
    elif len(init_tokens) < SEQ_LEN: init_tokens += [tokenizer.encode(" ", add_special_tokens=False)[0]] * (SEQ_LEN - len(init_tokens))
    initial_ids = torch.tensor(init_tokens, device=args.device).unsqueeze(0).repeat(16, 1)

    print(f"\nWarm Start Prompt: {tokenizer.decode(initial_ids[0])!r}")

    torch.cuda.empty_cache()
    print(f"\nStarting EPO (Ultra-Strict, Warm-Start)...")
    history = epo(
        runner, model, tokenizer, seq_len=SEQ_LEN, population_size=16, explore_per_pop=32, 
        iters=300, batch_size=256, # CRITICAL: Context-Aware must be batch_size=2 or 4 to avoid OOM
        restart_frequency=50, seed=42, initial_ids=initial_ids, **FLUENCY_PRESETS["ultra_strict"]
    )
    
    pareto = build_pareto_frontier(tokenizer, history)
    
    print("\n--- Evaluating Discovered Prompts ---")
    base_rate = eval_prompt_truthful(model, tokenizer, test_df, WARM_START_TEXT, mode="suffix")
    print(f"Baseline Hallucination Rate (with warm text): {base_rate:.2%}")
    
    results_data =[]
    for i, prompt in enumerate(pareto.text[:10]):
        rate = eval_prompt_truthful(model, tokenizer, test_df, prompt, mode="suffix")
        ppl = calculate_perplexity(model, tokenizer, prompt)
        print(f"\nPrompt {i}: {prompt!r}\nPPL: {ppl:.2f} | Hallucination: {rate:.2%} (Drop: {base_rate - rate:.2%})")
        results_data.append({"prompt": prompt, "perplexity": ppl, "hallucination_rate": rate, "delta": rate - base_rate})

    out_dir = f"results/ablations/fluent_discovery_hallu5_{args.method}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results_data, f, indent=2)

if __name__ == "__main__":
    main()