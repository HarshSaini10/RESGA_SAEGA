"""Contrastive persona-signature extraction.

A signature is the direction ``delta = mean(good) - mean(bad)`` in either the
residual stream (RESGA) or SAE-latent space (SAEGA), measured at the last token.
The optimizer then maximizes the projection onto this direction, which steers
generations toward the desired ("good") behavior and away from the persona.
"""
import json
import random

import torch
from tqdm import tqdm


def _contrastive_pair(row, template):
    """Build the ``(good_text, bad_text)`` pair for one example."""
    q = row["question"]
    if template == "answer_char":
        good = row["answer_not_matching_behavior"]
        bad = row["answer_matching_behavior"]
        if not good.startswith(" "):
            good = " " + good.strip()
        if not bad.startswith(" "):
            bad = " " + bad.strip()
        return f"{q}{good}", f"{q}{bad}"
    if template == "truthful_qa":
        true_ans = row["true_answer"]
        wrong_ans = random.choice(json.loads(row["incorrect_answers"]))
        return (
            f"Question: {q}\nAnswer: {true_ans}",
            f"Question: {q}\nAnswer: {wrong_ans}",
        )
    raise ValueError(f"Unknown signature template: {template}")


@torch.no_grad()
def extract_signature(model, tokenizer, sae, df, method, layer_idx, persona_cfg):
    """Return the RESGA delta vector, or the SAEGA ``(indices, weights)`` pair."""
    sig_cfg = persona_cfg["signature"]
    template = sig_cfg["template"]
    sample_n = sig_cfg["sample_n"]
    extract_df = df.sample(n=min(len(df), sample_n), random_state=42)

    print(f"Extracting {method} signature ({len(extract_df)} contrastive pairs)...")
    good_vecs, bad_vecs = [], []
    for _, row in tqdm(extract_df.iterrows(), total=len(extract_df)):
        good_text, bad_text = _contrastive_pair(row, template)

        out_good = model(**tokenizer(good_text, return_tensors="pt").to(model.device),
                         output_hidden_states=True)
        out_bad = model(**tokenizer(bad_text, return_tensors="pt").to(model.device),
                        output_hidden_states=True)

        resid_good = out_good.hidden_states[layer_idx + 1][0, -1, :]
        resid_bad = out_bad.hidden_states[layer_idx + 1][0, -1, :]

        if method == "sae":
            good_vecs.append(sae.encode(resid_good).cpu())
            bad_vecs.append(sae.encode(resid_bad).cpu())
        else:
            good_vecs.append(resid_good.cpu())
            bad_vecs.append(resid_bad.cpu())

    mu_good = torch.stack(good_vecs).mean(0).to(model.device)
    mu_bad = torch.stack(bad_vecs).mean(0).to(model.device)
    delta = mu_good - mu_bad

    if method == "residual":
        return delta

    # SAEGA: keep the top 1% most-discriminative latents and their signed weights.
    thresh = torch.quantile(delta.abs().float(), 0.99)
    indices = torch.where(delta.abs() > thresh)[0]
    return indices, delta[indices]
