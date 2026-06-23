"""
_summary_

Returns
-------
    _description_
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

from dreamy.epo import add_fwd_hooks
import os, yaml

ROOT = os.environ.get(
    "RESGA_SAEGA_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")
cfg = yaml.safe_load(open(CONFIG_PATH))
device = cfg['device']

def does_retokenize(model, tokenizer, input_ids):
    good = torch.empty(input_ids.shape[0], dtype=bool).to(device)
    input_strs = tokenizer.batch_decode(input_ids)
    for i, s in enumerate(input_strs):
        retokenized = tokenizer.encode(s, return_tensors="pt").to(device)
        if retokenized.shape[1] != input_ids.shape[1]:
            good[i] = False
        else:
            good[i] = (retokenized[0] == input_ids[i]).all()
        if not good[i]:
            print(f"bad input {i}: {s}")
    return good


def logit_diff_runner(
    model, tokenizer, token_id, banned_text, check_retokenization=False
):
    def run(input_ids=None, inputs_embeds=None):
        if input_ids is not None:
            if check_retokenization:
                good = does_retokenize(model, tokenizer, input_ids)
            else:
                good = torch.ones(input_ids.shape[0], dtype=bool).to(device)
            input_text = tokenizer.batch_decode(input_ids)
            good &= torch.tensor(
                [banned_text.lower() not in s.lower() for s in input_text], dtype=bool
            ).to(good.device)
        else:
            good = torch.ones(inputs_embeds.shape[0], dtype=bool).to(device)

        if input_ids is not None:
            output = model(input_ids)
        else:
            output = model(inputs_embeds=inputs_embeds)

        out = dict()
        out["logits"] = output.logits
        out["target"] = torch.where(
            good,
            output.logits[:, -1, token_id]
            - torch.where(
                output.logits[:, -1].argmax(dim=-1) == token_id,
                output.logits[:, -1].topk(dim=-1, k=2).values[:, 1],
                output.logits[:, -1].max(dim=-1).values,
            ),
            -torch.finfo(output.logits.dtype).max,
        )
        # probs = torch.log_softmax(last_logits, -1, :], dim=-1)
        # out["target"] = torch.where(
        #     good, probs[:, token_id], -torch.finfo(probs.dtype).max
        # )
        return out

    return run


def neuron_runner(model, tokenizer, layer, neuron, check_retokenization=False):
    def run(input_ids=None, inputs_embeds=None):
        if input_ids is not None:
            if check_retokenization:
                good = does_retokenize(model, tokenizer, input_ids)
            else:
                good = torch.ones(input_ids.shape[0], dtype=bool).to(device)
        else:
            good = torch.ones(inputs_embeds.shape[0], dtype=bool).to(device)

        out = {}

        def get_target(module, input, output):
            out["target"] = input[0][:, -1, neuron]

        hooks = [
            (model.gpt_neox.layers[layer].mlp.dense_4h_to_h, get_target),
        ]

        with add_fwd_hooks(hooks):
            if input_ids is not None:
                output = model(input_ids)
            else:
                output = model(inputs_embeds=inputs_embeds)

        out["logits"] = output.logits
        out["target"][~good] = -torch.finfo(out["target"].dtype).max
        return out

    return run


def residual_runner(model, tokenizer, layer, vector, check_retokenization=False):
    def run(input_ids=None, inputs_embeds=None):
        if input_ids is not None:
            if check_retokenization:
                good = does_retokenize(model, tokenizer, input_ids)
            else:
                good = torch.ones(input_ids.shape[0], dtype=bool).to(device)
        else:
            good = torch.ones(inputs_embeds.shape[0], dtype=bool).to(device)

        out = {}

        def get_target(module, input, output):
            resid = input[0][:, -1]
            std_resid = (resid - resid.mean(dim=-1, keepdim=True)) / resid.std(
                dim=-1, keepdim=True
            )
            out["target"] = std_resid @ vector

        hooks = [
            (model.gpt_neox.layers[layer], get_target),
        ]

        with add_fwd_hooks(hooks):
            if input_ids is not None:
                output = model(input_ids)
            else:
                output = model(inputs_embeds=inputs_embeds)

        out["logits"] = output.logits
        out["target"][~good] = -torch.finfo(out["target"].dtype).max
        return out

    return run


def attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor = None,
    position_ids: torch.LongTensor = None,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    has_layer_past = layer_past is not None

    # Compute QKV
    # Attention heads [batch, seq_len, hidden_size]
    #   --> [batch, seq_len, (np * 3 * head_size)]
    qkv = self.query_key_value(hidden_states)

    # [batch, seq_len, (num_heads * 3 * head_size)]
    #   --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
    query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

    # Compute rotary embeddings on rotary_ndims
    query_rot = query[..., : self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims :]
    key_rot = key[..., : self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims :]

    # Compute token offset for rotary embeddings (when decoding)
    seq_len = key.shape[-2]
    if has_layer_past:
        seq_len += layer_past[0].shape[-2]
    cos, sin = self.rotary_emb(value, seq_len=seq_len)
    query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
    query = torch.cat((query, query_pass), dim=-1)
    key = torch.cat((key, key_pass), dim=-1)

    # GPT-neo-X casts query and key in fp32 to apply rotary embedding in full precision
    target_dtype = value.dtype
    if query.dtype != target_dtype:
        query = query.to(target_dtype)
    if key.dtype != target_dtype:
        key = key.to(target_dtype)

    batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_length = key.size(-2)

    # dynamically increase the causal mask with the key length, if needed.
    if key_length > self.bias.shape[-1]:
        self._init_bias(key_length, device=key.device)
    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

    query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
    key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
    attn_scores = torch.zeros(
        batch_size * num_attention_heads,
        query_length,
        key_length,
        dtype=query.dtype,
        device=key.device,
    )
    attn_scores = torch.baddbmm(
        attn_scores,
        query,
        key.transpose(1, 2),
        beta=1.0,
        alpha=self.norm_factor,
    )
    attn_scores = attn_scores.view(
        batch_size, num_attention_heads, query_length, key_length
    )

    mask_value = torch.finfo(attn_scores.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(
        attn_scores.device
    )
    attn_scores = torch.where(causal_mask, attn_scores, mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        attn_scores = attn_scores + attention_mask

    attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

# def sycophancy_residual_runner(model, tokenizer, layer_idx, delta_resid):
#     delta_resid = delta_resid.to(
#     device=model.device,
#     dtype=next(model.parameters()).dtype,
# )

#     def run(input_ids=None, inputs_embeds=None):
#         if input_ids is not None:
#             outputs = model(
#                 input_ids,
#                 output_hidden_states=True,
#                 use_cache=False,
#             )
#         else:
#             outputs = model(
#                 inputs_embeds=inputs_embeds,
#                 output_hidden_states=True,
#                 use_cache=False,
#             )

#         resid = outputs.hidden_states[layer_idx][:, -1, :]
#         resid = (resid - resid.mean(dim=-1, keepdim=True)) / resid.std(dim=-1, keepdim=True)

#         # target = - (resid @ delta_resid).mean()
#         target = - (resid @ delta_resid)
#         return {
#             "logits": outputs.logits,
#             "target": target,
#         }

#     return run

# def sycophancy_sae_runner(
#     model,
#     tokenizer,
#     sae,
#     layer_idx,
#     latent_idx,
#     latent_sign,
# ):
#     latent_idx = latent_idx.to(model.device)
#     latent_sign = latent_sign.to(model.device)

#     def run(input_ids=None, inputs_embeds=None):
#         if input_ids is not None:
#             outputs = model(
#                 input_ids,
#                 output_hidden_states=True,
#                 use_cache=False,
#             )
#         else:
#             outputs = model(
#                 inputs_embeds=inputs_embeds,
#                 output_hidden_states=True,
#                 use_cache=False,
#             )

#         resid = outputs.hidden_states[layer_idx][:, -1, :]
#         resid = (resid - resid.mean(dim=-1, keepdim=True)) / resid.std(
#             dim=-1, keepdim=True
#         )

#         z = sae.encode(resid)                # [batch, n_latents]
#         z_sel = z[:, latent_idx]             # [batch, k]

#         target = -(z_sel * latent_sign).sum(dim=-1)

#         return {
#             "logits": outputs.logits,
#             "target": target,
#         }

#     return run

def sycophancy_residual_runner(model, tokenizer, layer_idx, delta_resid):
    """
    Optimizes inputs to minimize projection onto a dense residual direction.
    """
    # Ensure steering vector is on the correct device and dtype
    delta_resid = delta_resid.to(
        device=model.device,
        dtype=model.dtype
    )

    def run(input_ids=None, inputs_embeds=None):
        # 1. Run Model
        if input_ids is not None:
            outputs = model(input_ids, output_hidden_states=True, use_cache=False)
        else:
            outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False)

        # 2. Extract Residuals
        # HF Tuple: (embeddings, layer_1, ..., layer_N)
        # layer_idx maps to the output of that block.
        # We assume layer_idx is 0-indexed matching the config (e.g. 14).
        # We grab the last token's residual.
        resid = outputs.hidden_states[layer_idx + 1][:, -1, :]

        # CRITICAL FIX: Removed manual normalization ((resid-mean)/std). 
        # We use the raw residual stream to preserve magnitude semantics.

        # 3. Calculate Objective
        # We want to MINIMIZE similarity to the sycophancy vector.
        # EPO maximizes fitness, so we negate the dot product.
        target = - (resid @ delta_resid)

        return {
            "logits": outputs.logits,
            "target": target,
        }

    return run


def sycophancy_sae_runner(model, tokenizer, sae, layer_idx, latent_indices, latent_weights):
    """
    Optimizes inputs to minimize activation of specific SAE features, weighted by their 
    observed causal relevance (or correlation magnitude).
    """
    latent_indices = latent_indices.to(model.device)
    latent_weights = latent_weights.to(model.device, dtype=torch.float32)

    def run(input_ids=None, inputs_embeds=None):
        # 1. Run Model
        if input_ids is not None:
            outputs = model(input_ids, output_hidden_states=True, use_cache=False)
        else:
            outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False)

        # 2. Extract Residuals
        resid = outputs.hidden_states[layer_idx + 1][:, -1, :]

        # CRITICAL FIX: No normalization. SAE expects raw inputs.
        
        # 3. Encode with SAE
        # sae.encode returns the feature activations [batch_size, d_sae]
        feature_acts = sae.encode(resid)

        # 4. Filter & Weight
        # Select only the features we care about
        selected_acts = feature_acts[:, latent_indices] # [batch, k]
        
        # Calculate weighted sum
        # If weight is Positive (Pro-Sycophancy), we want to minimize Act.
        # If weight is Negative (Anti-Sycophancy), we want to maximize Act (minimize negative Act).
        # Therefore, we simply minimize Sum(Act * Weight).
        # EPO maximizes fitness, so we negate.
        target = - (selected_acts * latent_weights).sum(dim=-1)

        return {
            "logits": outputs.logits,
            "target": target,
        }

    return run


def hallucination_residual_runner(model, tokenizer, delta, layer_idx, train_prompts):
    """
    Minimizes hallucination by maximizing projection onto (Truth − Hallucination)
    using normalized residuals.
    """
    delta = delta.to(model.device, dtype=model.dtype)

    def run(input_ids=None, inputs_embeds=None):
        import random
        context_str = random.choice(train_prompts)
        ctx_ids = tokenizer(
            context_str,
            return_tensors="pt",
            add_special_tokens=True
        ).input_ids.to(model.device)

        pop_size = (
            input_ids.shape[0]
            if input_ids is not None
            else inputs_embeds.shape[0]
        )
        ctx_expanded = ctx_ids.repeat(pop_size, 1)

        if input_ids is not None:
            full_input_ids = torch.cat([ctx_expanded, input_ids], dim=1)
            outputs = model(
                full_input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        else:
            ctx_embeds = model.get_input_embeddings()(ctx_expanded)
            full_embeds = torch.cat([ctx_embeds, inputs_embeds], dim=1)
            outputs = model(
                inputs_embeds=full_embeds,
                output_hidden_states=True,
                use_cache=False,
            )

        # --- Residual (LAST token) ---
        resid = outputs.hidden_states[layer_idx + 1][:, -1, :]

        # ✅ REQUIRED: normalize residuals for hallucination geometry
        resid = (resid - resid.mean(dim=-1, keepdim=True)) / (
            resid.std(dim=-1, keepdim=True) + 1e-6
        )

        # ✅ Minimize hallucination = maximize Truth − Hallucination
        target = - (resid @ delta)

        # --- Return suffix logits only (for fluency constraint) ---
        suffix_start_idx = ctx_ids.shape[1] - 1
        suffix_logits = outputs.logits[:, suffix_start_idx:-1, :]

        return {
            "logits": suffix_logits,
            "target": target,
        }

    return run

def myopic_residual_runner(model, tokenizer, delta, layer_idx, train_prompts):
    """
    Reduce myopia by minimizing projection onto (Myopic − Non-Myopic).
    """
    delta = delta.to(model.device, dtype=model.dtype)

    def run(input_ids=None, inputs_embeds=None):
        import random
        ctx = random.choice(train_prompts)
        ctx_ids = tokenizer(ctx, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)

        pop = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        ctx_ids = ctx_ids.repeat(pop, 1)

        if input_ids is not None:
            full_ids = torch.cat([ctx_ids, input_ids], dim=1)
            out = model(full_ids, output_hidden_states=True, use_cache=False)
        else:
            ctx_emb = model.get_input_embeddings()(ctx_ids)
            full_emb = torch.cat([ctx_emb, inputs_embeds], dim=1)
            out = model(inputs_embeds=full_emb, output_hidden_states=True, use_cache=False)

        resid = out.hidden_states[layer_idx + 1][:, -1, :]

        # REQUIRED normalization
        resid = (resid - resid.mean(dim=-1, keepdim=True)) / (
            resid.std(dim=-1, keepdim=True) + 1e-6
        )

        # Minimize myopia
        target = - (resid @ delta)

        start = ctx_ids.shape[1] - 1
        logits = out.logits[:, start:-1]

        return {"logits": logits, "target": target}

    return run
