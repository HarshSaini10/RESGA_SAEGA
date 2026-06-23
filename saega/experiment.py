"""End-to-end RESGA / SAEGA run for a single persona configuration."""
import json
import os

import torch

from dreamy.epo import epo, build_pareto_frontier

from . import config as cfgmod
from .data import context_samples, load_split
from .evaluation import evaluate
from .models import load_model, load_sae, perplexity
from .runners import build_runner
from .signatures import extract_signature


def run_persona(persona, method, fluency, layer_idx, device,
                model_name=None, sae_release=None, hook_point=None,
                run_name=None, top_k=5):
    """Run one persona × method × fluency × layer configuration and save metrics."""
    cfg = cfgmod.load_config()
    persona_cfg = cfgmod.get_persona(persona)

    model_name = model_name or cfg["model"]["name"]
    cache_dir = os.environ.get("HF_HOME", cfg["model"]["cache_dir"])
    run_name = run_name or f"{persona}_{method}_l{layer_idx}_{fluency}"
    if method == "sae" and not (sae_release and hook_point):
        raise ValueError("SAEGA (--method sae) requires --sae_release and --hook_point")

    model, tokenizer = load_model(model_name, device, cache_dir, cfg["model"]["dtype"])
    sae = load_sae(sae_release, hook_point, device) if method == "sae" else None

    train_df, test_df = load_split(persona_cfg)
    ctx = context_samples(train_df, persona_cfg)

    signature = extract_signature(model, tokenizer, sae, train_df, method, layer_idx, persona_cfg)
    runner = build_runner(model, tokenizer, method, signature, layer_idx,
                          persona_cfg["runner"], ctx, sae=sae)

    torch.cuda.empty_cache()
    print(f"Starting EPO: {run_name} ({fluency})")
    history = epo(
        runner, model, tokenizer,
        seed=cfg["seed"],
        **cfg["epo"],
        **cfg["fluency_presets"][fluency],
    )
    pareto = build_pareto_frontier(tokenizer, history)

    eval_cfg = persona_cfg["eval"]
    metric = eval_cfg["metric_name"]
    print("Evaluating Pareto frontier...")
    baseline = evaluate(model, tokenizer, test_df, "", eval_cfg)
    base_rate = baseline[metric]
    print(f"Baseline {metric}: {base_rate:.2%}")

    results = []
    for prompt in pareto.text[:top_k]:
        stats = evaluate(model, tokenizer, test_df, prompt, eval_cfg)
        entry = {
            "prompt": prompt,
            "perplexity": perplexity(model, tokenizer, prompt),
            metric: stats[metric],
            "baseline_" + metric: base_rate,
            "delta": stats[metric] - base_rate,
        }
        if stats.get("mask") is not None and baseline.get("mask") is not None:
            entry["becomes_target"] = int((~baseline["mask"] & stats["mask"]).sum())
            entry["leaves_target"] = int((baseline["mask"] & ~stats["mask"]).sum())
            entry["mean_logprob_a"] = stats["mean_logprob_a"]
            entry["mean_logprob_b"] = stats["mean_logprob_b"]
        results.append(entry)
        print(f"  {prompt!r} | {metric}={stats[metric]:.2%} | ppl={entry['perplexity']:.1f}")

    out_dir = cfgmod.resolve_path(os.path.join(persona_cfg["out_dir"], run_name))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")
    return results
