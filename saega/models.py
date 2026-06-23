"""Model, tokenizer and SAE loading, plus perplexity."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_model(model_name, device, cache_dir=None, dtype="float32"):
    """Load a frozen causal LM and its tokenizer (left-padded, eos as pad)."""
    torch_dtype = _DTYPES.get(dtype, dtype if isinstance(dtype, torch.dtype) else torch.float32)
    print(f"Loading {model_name} on {device} ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device, cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def load_sae(sae_release, hook_point, device):
    """Load a frozen SAE in float32.

    Handles both sae_lens API variants (``SAE.from_pretrained`` returning either
    the SAE directly or a ``(sae, cfg, sparsity)`` tuple).
    """
    print(f"Loading SAE {sae_release} ({hook_point})...")
    result = SAE.from_pretrained(sae_release, sae_id=hook_point, device=device)
    sae = result[0] if isinstance(result, (tuple, list)) else result
    sae = sae.to(dtype=torch.float32)
    for p in sae.parameters():
        p.requires_grad = False
    return sae


@torch.no_grad()
def perplexity(model, tokenizer, text):
    if not text:
        return 0.0
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model(**inputs, labels=inputs.input_ids)
    return torch.exp(out.loss).item()
