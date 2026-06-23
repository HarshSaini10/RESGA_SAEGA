#!/usr/bin/env python3
"""Unified RESGA / SAEGA driver.

Runs fluent gradient ascent for one persona, method, fluency and layer.
Replaces the former per-persona scripts (sycophancy / myopic / hallucination).

Examples
--------
RESGA on sycophancy (residual stream, layer 14):

    python main.py --persona sycophancy --method residual \\
        --fluency medium --layer_idx 14 --device cuda:0

SAEGA on hallucination (needs an SAE release + hook point):

    python main.py --persona hallucination --method sae \\
        --fluency medium --layer_idx 25 --device cuda:0 \\
        --model_name meta-llama/Llama-3.1-8B-Instruct \\
        --sae_release Juliushanhanhan/llama-3-8b-it-res \\
        --hook_point blocks.25.hook_resid_post
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from saega.experiment import run_persona


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--persona", required=True, choices=["sycophancy", "myopic", "hallucination"])
    p.add_argument("--method", required=True, choices=["residual", "sae"],
                   help="residual = RESGA, sae = SAEGA")
    p.add_argument("--fluency", required=True, choices=["strict", "medium", "loose"])
    p.add_argument("--layer_idx", type=int, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--model_name", default=None)
    p.add_argument("--sae_release", default=None)
    p.add_argument("--hook_point", default=None)
    p.add_argument("--run_name", default=None)
    args = p.parse_args()

    run_persona(
        persona=args.persona, method=args.method, fluency=args.fluency,
        layer_idx=args.layer_idx, device=args.device, model_name=args.model_name,
        sae_release=args.sae_release, hook_point=args.hook_point, run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
