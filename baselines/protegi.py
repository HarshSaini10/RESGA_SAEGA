#!/usr/bin/env python3
"""ProTeGi baseline: LLM-as-optimizer prompt search.

An optimizer LLM (via OpenRouter) proposes prompt edits from the target model's
mistakes; candidates are beam-searched against the target model. Set the
``OPENROUTER_API_KEY`` environment variable before running.

    OPENROUTER_API_KEY=... python baselines/protegi.py --device cuda:0 \\
        --model_name meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import json
import os
import random
import time

import httpx
import torch
from tqdm import tqdm

import common

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPTIMIZER_MODEL = "openai/gpt-4o"


def chatgpt(prompt, n=1, temperature=0.7):
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    payload = {"model": OPTIMIZER_MODEL, "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "n": n}
    for _ in range(5):
        try:
            r = httpx.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}",
                                             "Content-Type": "application/json"},
                           json=payload, timeout=60.0)
            r.raise_for_status()
            return [c["message"]["content"] for c in r.json()["choices"]]
        except Exception as e:
            print(f"API Error: {e}. Retrying...")
            time.sleep(2)
    return []


def parse_tagged_text(text, start="<START>", end="<END>"):
    out = []
    while start in text and end in text[text.find(start):]:
        s = text.find(start) + len(start)
        e = text.find(end, s)
        out.append(text[s:e].strip())
        text = text[e + len(end):]
    return out


@torch.no_grad()
def evaluate_with_errors(model, tokenizer, df, prompt, persona, collect=False):
    """Return (behavior_rate, error_examples). An error = persona exhibited."""
    from saega.evaluation import evaluate, qa_logprob
    import json as _json

    if persona == "hallucination":
        errors, n = [], 0
        for _, row in df.iterrows():
            q = f"{row['question']} {prompt}".strip()
            lp_true = qa_logprob(model, tokenizer, [q], [row["true_answer"]]).item()
            wrong = _json.loads(row["incorrect_answers"])
            lp_false = qa_logprob(model, tokenizer, [q] * len(wrong), wrong).max().item()
            if lp_false > lp_true:
                n += 1
                if collect:
                    errors.append(f"Question: {row['question']}\nModel preferred a hallucination.")
        return n / len(df), errors

    stats = evaluate(model, tokenizer, df, prompt, common.TASKS[persona]["eval"])
    mask = stats["mask"]
    errors = []
    if collect:
        questions = df["question"].tolist()
        errors = [f"Question: {questions[i]}\nModel exhibited {persona}."
                  for i in range(len(mask)) if mask[i]]
    return stats["rate"], errors


class ProTeGi:
    def __init__(self, model, tokenizer, persona, train_df):
        self.model, self.tokenizer, self.persona, self.train_df = model, tokenizer, persona, train_df
        self.n_gradients, self.errors_per_gradient, self.steps_per_gradient = 2, 4, 2

    def _gradients(self, prompt, error_examples):
        sample = random.sample(error_examples, min(len(error_examples), self.errors_per_gradient))
        joined = "\n\n".join(f"Example {i+1}:\n{e}" for i, e in enumerate(sample))
        res = chatgpt(f'I am writing a short prefix appended to questions to stop a model '
                      f'from exhibiting {self.persona}.\n\nMy current prefix is:\n"{prompt}"\n\n'
                      f'But it fails on these examples:\n{joined}\n\nGive 2 distinct reasons why '
                      f'it failed. Wrap each reason with <START> and <END>.')
        return parse_tagged_text(res[0]) if res else []

    def _apply(self, prompt, feedback):
        res = chatgpt(f'I am writing a short prefix to prevent {self.persona}.\nMy current prefix '
                      f'is:\n"{prompt}"\n\nThe problem is: {feedback}\n\nWrite {self.steps_per_gradient} '
                      f'new, improved, short prefixes (under 10 words). Wrap each with <START> and <END>.')
        return parse_tagged_text(res[0]) if res else []

    def expand(self, prompts):
        minibatch = self.train_df.sample(n=min(len(self.train_df), 32))
        new = []
        for prompt in tqdm(prompts, desc="Expanding"):
            _, errors = evaluate_with_errors(self.model, self.tokenizer, minibatch, prompt,
                                             self.persona, collect=True)
            if not errors:
                continue
            feedbacks = [f for _ in range(self.n_gradients) for f in self._gradients(prompt, errors)]
            for fb in feedbacks:
                new.extend(self._apply(prompt, fb))
        return list(set(prompts + [p.strip() for p in new if p.strip()]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", required=True)
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--personas", nargs="+", default=["sycophancy", "myopic", "hallucination"])
    args = p.parse_args()

    model, tokenizer = common.load_model(args.model_name, args.device)
    results = []
    for persona in args.personas:
        print(f"\n{'='*50}\nProTeGi: {persona}\n{'='*50}")
        train_df, test_df = common.load_task(persona)
        base = common.behavior_rate(model, tokenizer, test_df, "", persona)
        print(f"Zero-shot baseline: {base:.2%}")

        opt = ProTeGi(model, tokenizer, persona, train_df)
        beam = ["Please think carefully and answer honestly."]
        for r in range(args.rounds):
            print(f"\n--- Round {r+1}/{args.rounds} | beam={beam} ---")
            candidates = opt.expand(beam)
            subset = train_df.sample(n=min(len(train_df), 128))
            scored = sorted(((common.behavior_rate(model, tokenizer, subset, c, persona), c)
                             for c in tqdm(candidates, desc="Scoring")), key=lambda x: x[0])
            beam = [c for _, c in scored[:args.beam_size]]
            print(f"Best this round: {scored[0][0]:.2%} -> {scored[0][1]!r}")

        final = common.behavior_rate(model, tokenizer, test_df, beam[0], persona)
        print(f"Final test rate: {final:.2%} (delta {final - base:.2%})")
        results.append({"persona": persona, "baseline": base, "final_rate": final,
                        "delta": final - base, "best_prompt": beam[0], "beam": beam})

    out_dir = f"results/baselines/protegi/{args.model_name.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "protegi_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir}/protegi_results.json")


if __name__ == "__main__":
    main()
