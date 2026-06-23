# RESGA / SAEGA

Code for the TMLR paper **"Bridging Mechanistic Interpretability and Prompt
Engineering with Gradient Ascent for Interpretable Persona Control"**.

- Paper (OpenReview): https://openreview.net/forum?id=dcmHPxgo4c
- Preprint (arXiv): https://arxiv.org/abs/2601.02896

We adapt **fluent gradient ascent** to optimize randomly-initialized prompts so
that a model's internal representation aligns with (or against) an identified
*persona direction*. Two variants:

- **RESGA** (`--method residual`): gradient ascent against a persona direction
  in the **residual stream**.
- **SAEGA** (`--method sae`): gradient ascent against a sparse set of
  interpretable **SAE latents** for the same persona.

Personas: **sycophancy**, **hallucination**, **myopic reward** — across
**Llama-3.1-8B-Instruct**, **Qwen2.5-7B-Instruct**, and **Gemma-3-4b-it**.

## Repository layout

```
run.sh             Single entry point (main | sweep | baseline)
main.py            Unified RESGA/SAEGA driver (one persona run)
sweep.py           Config-driven multi-GPU sweep launcher
configs/
  config.yaml      Model, cache, EPO hyperparameters, fluency presets
  personas.yaml    Per-persona dataset / signature / runner / eval settings
  sweep.yaml       Sweep axes (models x personas x methods x fluencies x layers)
saega/             Shared library
  config.py  models.py  data.py  signatures.py  runners.py  evaluation.py  experiment.py
dreamy/            Vendored, modified fork of the fluent-dreaming library (EPO engine)
baselines/         common.py + gcg, protegi, prefix_tuning, prompt_steering
analysis/          Result analysis, causal-latent discovery, sweeps, data prep
ablations/         Appendix experiments (random direction/updates, k/layer/length sweeps, ...)
data/              See data/README.md — raw datasets are NOT committed
```

### How a persona run works

`main.py` calls `saega.experiment.run_persona`, which:

1. extracts a **signature** `delta = mean(good) - mean(bad)` at the chosen layer
   (residual vector for RESGA, top-1% SAE latents for SAEGA);
2. builds a **runner** that maximizes the projection of the prompt's last-token
   representation onto `delta`;
3. runs **fluent gradient ascent** (`dreamy.epo`) with the configured fluency
   preset, builds the Pareto frontier, and
4. scores the top prompts with the persona's metric, writing `metrics.json`.

All three personas share this path; they differ only via `configs/personas.yaml`
(dataset, contrastive template, whether a question context is prepended, and the
evaluation metric). The objective is mathematically identical across personas,
so this consolidation preserves the original numerics.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- A CUDA-enabled PyTorch build matching your system is assumed.
- `dictionary-learning` (used by some SAE ablations) may need installing from
  source: `pip install git+https://github.com/saprmarks/dictionary_learning.git`.
- Models/SAEs download to `config.yaml:model.cache_dir` (default `./hf_cache`) or
  `HF_HOME`.

## Data

Raw datasets are not redistributed — see [`data/README.md`](data/README.md) for
sources and `python analysis/prep_truthfulqa.py` to regenerate TruthfulQA.

## Running

Everything goes through `run.sh` (flags pass straight through):

```bash
# RESGA on sycophancy
./run.sh main --persona sycophancy --method residual \
    --fluency medium --layer_idx 14 --device cuda:0

# SAEGA on hallucination (needs an SAE)
./run.sh main --persona hallucination --method sae \
    --fluency medium --layer_idx 25 --device cuda:0 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --sae_release Juliushanhanhan/llama-3-8b-it-res \
    --hook_point blocks.25.hook_resid_post

# Full sweep (edit configs/sweep.yaml first; --dry_run to preview)
./run.sh sweep --dry_run
./run.sh sweep

# Baselines
./run.sh baseline gcg             --device cuda:0
./run.sh baseline protegi         --device cuda:0     # set OPENROUTER_API_KEY first
./run.sh baseline prefix_tuning   --device cuda:0
./run.sh baseline prompt_steering --device cuda:0 --layer_idx 14
```

Persona run outputs land in `results/runs/<persona>/<run_name>/metrics.json`;
baseline outputs in `results/baselines/<name>/`.

## Baselines

All baselines load the target model and score prompts through the **same**
metric as the main method (`saega.evaluation`) for a fair comparison:

- `baselines/gcg.py` — GCG-style discrete optimization toward the desired answer.
- `baselines/protegi.py` — ProTeGi (an LLM proposes prompt edits via OpenRouter).
- `baselines/prefix_tuning.py` — continuous soft-prompt tuning.
- `baselines/prompt_steering.py` — zero-shot / instruction / random prompts and
  activation steering (consolidates the former per-persona baseline scripts).

## Configuration

`configs/config.yaml` holds model defaults, the shared EPO hyperparameters, and
fluency presets. `configs/personas.yaml` defines each persona. Paths resolve
relative to the repo root; override it with the `RESGA_SAEGA_ROOT` env var.

## Citation

```bibtex
@article{saini2026bridging,
  title={Bridging Mechanistic Interpretability and Prompt Engineering with Gradient Ascent for Interpretable Persona Control},
  author={Saini, Harshvardhan and Tang, Yiming and Liu, Dianbo},
  journal={arXiv preprint arXiv:2601.02896},
  year={2026}
}
```

## Acknowledgements & license

Released under the MIT License (see [`LICENSE`](LICENSE)). Builds on a modified
fork of [`dreamy`](https://github.com/Confirm-Solutions/dreamy) (Confirm Labs,
MIT; original license at `dreamy/LICENSE`). Datasets are © their respective
authors (Anthropic evals, TruthfulQA, HaluEval).
