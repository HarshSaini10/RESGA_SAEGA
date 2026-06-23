#!/usr/bin/env python3
"""Config-driven multi-GPU sweep launcher.

Expands the axes in a sweep config (default: configs/sweep.yaml) into one
``main.py`` invocation per combination and runs them round-robin across the
listed GPUs (one worker process per GPU). Replaces the former per-experiment
orchestrator scripts.

    python sweep.py                       # uses configs/sweep.yaml
    python sweep.py --config my_sweep.yaml
    python sweep.py --dry_run             # print the planned runs only
"""
import argparse
import os
import subprocess
from itertools import product
from multiprocessing import Process

import yaml

os.environ.setdefault("HF_HUB_DISABLE_HTTP2", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def build_tasks(cfg):
    tasks = []
    for model in cfg["models"]:
        axes = product(cfg["personas"], cfg["methods"], cfg["fluencies"], cfg["layers"])
        for persona, method, fluency, layer in axes:
            run_name = f"{model['tag']}_{persona}_{method}_l{layer}_{fluency}"
            cmd = [
                "python", "main.py",
                "--persona", persona,
                "--method", method,
                "--fluency", fluency,
                "--layer_idx", str(layer),
                "--model_name", model["name"],
                "--run_name", run_name,
            ]
            if method == "sae":
                cmd += ["--sae_release", model["sae_release"],
                        "--hook_point", model["sae_hook_point"]]
            else:
                cmd += ["--hook_point", f"blocks.{layer}.hook_resid_post"]
            tasks.append((run_name, cmd))
    return tasks


def run_queue(gpu, tasks):
    os.makedirs("results/logs", exist_ok=True)
    for run_name, cmd in tasks:
        print(f"[{gpu}] launching {run_name}")
        with open(f"results/logs/{run_name}.log", "w") as f:
            subprocess.run(cmd + ["--device", gpu], stdout=f, stderr=subprocess.STDOUT)
        print(f"[{gpu}] finished {run_name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sweep.yaml")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tasks = build_tasks(cfg)
    gpus = cfg["gpus"]

    print(f"{len(tasks)} run(s) across {len(gpus)} GPU(s).")
    if args.dry_run:
        for run_name, cmd in tasks:
            print(run_name, ":", " ".join(cmd))
        return

    queues = {gpu: [] for gpu in gpus}
    for i, task in enumerate(tasks):
        queues[gpus[i % len(gpus)]].append(task)

    procs = [Process(target=run_queue, args=(gpu, q)) for gpu, q in queues.items() if q]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    print("Sweep complete.")


if __name__ == "__main__":
    main()
