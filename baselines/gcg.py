#!/usr/bin/env python3
"""GCG-style discrete-prompt baseline.

Optimizes a prompt to directly maximize log P(desired answer) using the same
fluent gradient-ascent engine (``dreamy.epo``) as the main method, but with a
behavioral target instead of a representation target. Scoring reuses the shared
persona metric.

    python baselines/gcg.py --device cuda:0 --fluency loose \\
        --model_name meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import json
import os
import random

import torch

import common
from saega.models import perplexity
from dreamy.epo import epo, build_pareto_frontier

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# GCG runs with essentially no fluency constraint by default.
FLUENCY_PRESETS = {
    "strict": {"x_penalty_min": 2.0, "x_penalty_max": 12.0, "restart_xentropy": 3.0, "restart_xentropy_max_mult": 1.5},
    "medium": {"x_penalty_min": 0.5, "x_penalty_max": 4.0, "restart_xentropy": 1.5, "restart_xentropy_max_mult": 2.0},
    "loose": {"x_penalty_min": 1e-5, "x_penalty_max": 1e-5, "restart_xentropy": 0.5, "restart_xentropy_max_mult": 2.0},
}


def behavioral_runner(model, tokenizer, train_samples, persona):
    """EPO runner that maximizes log P(desired answer | context + prompt)."""
    def run(input_ids=None, inputs_embeds=None):
        row = random.choice(train_samples)
        q = row["question"]
        if persona == "hallucination":
            ctx_text, ans_text = f"Question: {q}\nAnswer:", row["true_answer"]
        else:  # sycophancy / myopic: target the non-matching (good) letter
            bad = row["answer_matching_behavior"].strip()
            ans_text = " (B)" if bad == "(A)" else " (A)"
            ctx_text = q
        if not ans_text.startswith(" "):
            ans_text = " " + ans_text.strip()

        ctx = tokenizer(ctx_text, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        ans = tokenizer(ans_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        pop = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        ctx, ans = ctx.repeat(pop, 1), ans.repeat(pop, 1)

        if input_ids is not None:
            full = torch.cat([ctx, input_ids, ans], dim=1)
            outputs = model(full, use_cache=False)
            shift_labels = full[:, 1:].contiguous()
            prompt_len = input_ids.shape[1]
        else:
            emb = model.get_input_embeddings()
            full_emb = torch.cat([emb(ctx), inputs_embeds, emb(ans)], dim=1)
            outputs = model(inputs_embeds=full_emb, use_cache=False)
            prompt_len = inputs_embeds.shape[1]
            shift_labels = torch.cat([
                torch.zeros_like(ctx),
                torch.zeros(pop, prompt_len, dtype=torch.long, device=model.device),
                ans,
            ], dim=1)[:, 1:]

        ans_start = ctx.shape[1] + prompt_len - 1
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        losses = torch.nn.CrossEntropyLoss(reduction="none")(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        ).view(shift_labels.size())
        mask = torch.zeros_like(shift_labels).float()
        mask[:, ans_start:] = 1.0
        target = -(losses * mask).sum(dim=1)  # maximize log P(answer)

        suffix = outputs.logits[:, ctx.shape[1] - 1: ctx.shape[1] + prompt_len - 1, :]
        return {"logits": suffix, "target": target}
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", required=True)
    p.add_argument("--fluency", choices=["strict", "medium", "loose"], default="loose")
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--personas", nargs="+", default=["hallucination"])
    args = p.parse_args()

    model, tokenizer = common.load_model(args.model_name, args.device)
    results = []
    for persona in args.personas:
        print(f"\n{'='*50}\nGCG: {persona}\n{'='*50}")
        train_df, test_df = common.load_task(persona)
        train_samples = train_df.iloc[:32].to_dict("records")
        base = common.behavior_rate(model, tokenizer, test_df, "", persona)
        print(f"Baseline rate: {base:.2%}")

        torch.cuda.empty_cache()
        history = epo(behavioral_runner(model, tokenizer, train_samples, persona),
                      model, tokenizer, seq_len=8, population_size=16, explore_per_pop=32,
                      iters=10, batch_size=64, restart_frequency=50, seed=42,
                      **FLUENCY_PRESETS[args.fluency])
        pareto = build_pareto_frontier(tokenizer, history)

        prompts = []
        for prompt in pareto.text[:3]:
            rate = common.behavior_rate(model, tokenizer, test_df, prompt, persona)
            prompts.append({"prompt": prompt, "rate": rate, "delta": rate - base,
                            "perplexity": perplexity(model, tokenizer, prompt)})
            print(f"  {prompt!r} | rate={rate:.2%}")
        results.append({"persona": persona, "baseline": base, "gcg_results": prompts})

    out_dir = f"results/baselines/gcg/{args.model_name.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"gcg_{args.fluency}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
