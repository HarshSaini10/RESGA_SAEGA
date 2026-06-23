"""Unified gradient-ascent runners for RESGA and SAEGA.

A runner is the ``cache_run`` callable consumed by ``dreamy.epo``. On each call
it runs the model on the candidate prompt (optionally prepended with a sampled
question context), reads the last-token representation, and returns:

* ``target``  -- the fitness EPO maximizes: projection onto the signature
                 ``delta = good - bad`` (RESGA) or the weighted SAE activations
                 (SAEGA). Higher = more aligned with the desired behavior.
* ``logits``  -- the logits used for the fluency (cross-entropy) penalty.

Personas differ only via three knobs taken from ``personas.yaml``:
``context_template`` (prepend a question or not), ``normalize`` (z-score the
residual), and ``full_logits`` (penalize the whole sequence vs. the suffix only).
"""
import random

import torch


def _forward(model, tokenizer, layer_idx, input_ids, inputs_embeds,
             context_template, context_samples, normalize):
    """Run the model (optionally with a sampled context prefix).

    Returns ``(last_token_representation, outputs, suffix_start)`` where
    ``suffix_start`` is ``None`` when no context is prepended.
    """
    if context_template:
        ctx_text = context_template.format(**random.choice(context_samples))
        ctx_ids = tokenizer(ctx_text, return_tensors="pt",
                            add_special_tokens=True).input_ids.to(model.device)
        pop = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        ctx = ctx_ids.repeat(pop, 1)
        if input_ids is not None:
            outputs = model(torch.cat([ctx, input_ids], dim=1),
                            output_hidden_states=True, use_cache=False)
        else:
            ctx_emb = model.get_input_embeddings()(ctx)
            outputs = model(inputs_embeds=torch.cat([ctx_emb, inputs_embeds], dim=1),
                            output_hidden_states=True, use_cache=False)
        suffix_start = ctx_ids.shape[1] - 1
    else:
        seq = input_ids if input_ids is not None else inputs_embeds
        kw = {"input_ids" if input_ids is not None else "inputs_embeds": seq}
        outputs = model(**kw, output_hidden_states=True, use_cache=False)
        suffix_start = None

    resid = outputs.hidden_states[layer_idx + 1][:, -1, :]
    if normalize:
        resid = (resid - resid.mean(dim=-1, keepdim=True)) / (resid.std(dim=-1, keepdim=True) + 1e-6)
    return resid, outputs, suffix_start


def _fluency_logits(outputs, suffix_start, full_logits):
    if full_logits or suffix_start is None:
        return outputs.logits
    return outputs.logits[:, suffix_start:-1, :]


def build_runner(model, tokenizer, method, signature, layer_idx, runner_cfg,
                 context_samples, sae=None):
    """Create the EPO runner for a persona/method.

    ``signature`` is the residual delta (RESGA) or ``(indices, weights)`` (SAEGA).
    """
    context_template = runner_cfg.get("context_template")
    normalize = runner_cfg.get("normalize", False)
    full_logits = runner_cfg.get("full_logits", False)

    if method == "residual":
        delta = signature.to(model.device, dtype=model.dtype)

        def run(input_ids=None, inputs_embeds=None):
            resid, outputs, suffix_start = _forward(
                model, tokenizer, layer_idx, input_ids, inputs_embeds,
                context_template, context_samples, normalize)
            target = resid @ delta
            return {"logits": _fluency_logits(outputs, suffix_start, full_logits), "target": target}

    elif method == "sae":
        indices, weights = signature
        indices = indices.to(model.device)
        weights = weights.to(model.device, dtype=torch.float32)

        def run(input_ids=None, inputs_embeds=None):
            resid, outputs, suffix_start = _forward(
                model, tokenizer, layer_idx, input_ids, inputs_embeds,
                context_template, context_samples, normalize)
            acts = sae.encode(resid)[:, indices]
            target = (acts.float() * weights).sum(dim=-1)
            return {"logits": _fluency_logits(outputs, suffix_start, full_logits), "target": target}

    else:
        raise ValueError(f"Unknown method: {method}")

    return run
