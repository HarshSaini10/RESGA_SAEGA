#!/usr/bin/env python3
"""Prompt-engineering and activation-steering baselines.

For each persona this reports the behavior rate (lower = better) under:

* zero-shot (no prompt),
* a hand-written instruction prompt,
* a random-token prompt (control),
* activation steering: add ``coef * (good - bad)`` direction to a layer's
  residual stream and report the best coefficient.

Consolidates the former per-persona prompt/steering baseline scripts. The
steering direction reuses ``saega.signatures`` and scoring reuses
``saega.evaluation`` (same metric as the main method).

    python baselines/prompt_steering.py --device cuda:0 --layer_idx 14 \\
        --model_name meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import json
import os
import random

import torch

import common
from saega.config import get_persona
from saega.data import load_split
from saega.signatures import extract_signature

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

RANDOM_PROMPT = "Blue migration texture seventeen logic barrier clouds."
STEER_COEFS = [-50, -20, -10, 10, 20, 50]


def find_layers(model):
    """Locate the transformer block ModuleList across architectures."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    # Fallback: largest ModuleList in the model.
    return max((m for _, m in model.named_modules() if isinstance(m, torch.nn.ModuleList)),
               key=len)


class SteeringHook:
    def __init__(self, model, layer_idx, vector):
        self.layer = find_layers(model)[layer_idx]
        self.vector = vector
        self.coef = 0.0
        self.handle = None

    def _hook(self, module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hidden = hidden + self.coef * self.vector.to(hidden.device, hidden.dtype)
        return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def run_persona(model, tokenizer, persona, layer_idx):
    persona_cfg = get_persona(persona)
    train_df, test_df = load_split(persona_cfg)
    instruction = common.TASKS[persona]["instruction"]

    rates = {
        "zero_shot": common.behavior_rate(model, tokenizer, test_df, "", persona),
        "instruction_prompt": common.behavior_rate(model, tokenizer, test_df, instruction, persona),
        "random_prompt": common.behavior_rate(model, tokenizer, test_df, RANDOM_PROMPT, persona),
    }

    # Activation steering along the good-bad direction.
    delta = extract_signature(model, tokenizer, None, train_df, "residual", layer_idx, persona_cfg)
    delta = delta / delta.norm()
    hook = SteeringHook(model, layer_idx, delta)
    best = min_rate = float("inf")
    for coef in STEER_COEFS:
        hook.coef = coef
        with hook:
            rate = common.behavior_rate(model, tokenizer, test_df, "", persona)
        if rate < min_rate:
            min_rate, best = rate, coef
    rates["best_steering"] = min_rate
    rates["best_steering_coef"] = best
    return rates


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", required=True)
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--layer_idx", type=int, default=25)
    p.add_argument("--personas", nargs="+", default=["sycophancy", "myopic", "hallucination"])
    args = p.parse_args()

    random.seed(42)
    model, tokenizer = common.load_model(args.model_name, args.device)

    results = {}
    for persona in args.personas:
        print(f"\n{'='*50}\n{persona}\n{'='*50}")
        results[persona] = run_persona(model, tokenizer, persona, args.layer_idx)
        for k, v in results[persona].items():
            print(f"  {k}: {v:.2%}" if isinstance(v, float) else f"  {k}: {v}")

    out_dir = f"results/baselines/prompt_steering/{args.model_name.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir}/results.json")


if __name__ == "__main__":
    main()
