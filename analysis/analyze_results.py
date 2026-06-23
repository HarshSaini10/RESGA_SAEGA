#!/usr/bin/env python3
import os
import json
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# =====================================================
# Configuration & Style
# =====================================================
TASK = "Sycophancy"
RESULTS_DIR = "results/runs"
DATA_PATH = "data/sycophancy.csv"
METRICS_CSV = "results/analysis/consolidated_results.csv"

# Model Config
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_RELEASE = "Juliushanhanhan/llama-3-8b-it-res"
HOOK_POINT = "blocks.25.hook_resid_post"
LAYER_IDX = 25
DEVICE = "cuda:0"
CACHE_DIR = os.environ.get("HF_HOME", "./hf_cache")

# Dense Steering Config
DENSE_COEF = -15.0

# Output
OUT_DIR = "results/paper_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Plotting Style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 2,
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 15,
    'lines.linewidth': 3,
    'lines.markersize': 8,
    'figure.dpi': 500,
    'savefig.dpi': 500,
    'font.family': 'sans-serif',
})

LABELS = {
    "Base": "Baseline",
    "Dense": "Steering Vector",
    "Resid": "RESGA (Ours)",
    "Sae": "SAEGA (Ours)"
}
PALETTE = {
    "Base": "#95a5a6", # Grey
    "Dense": "#3498db", # Blue
    "Resid": "#e74c3c", # Red
    "Sae": "#2ecc71",    # Green
    "Baseline": "#95a5a6", # Grey
    "Steering Vector": "#3498db", # Blue
    "RESGA (Ours)": "#e74c3c", # Red
    "SAEGA (Ours)": "#2ecc71"    # Green
}

# =====================================================
# 1. Helpers
# =====================================================
def get_best_prompts(results_dir):
    print(f"Scanning {results_dir} for best prompts...")
    best_sae = {"score": 1.0, "prompt": None}
    best_resid = {"score": 1.0, "prompt": None}
    
    subdirs = glob.glob(os.path.join(results_dir, "*"))
    for folder in subdirs:
        if not os.path.isdir(folder): continue
        name = os.path.basename(folder).lower()
        if "qwen" in name: continue
        
        json_path = os.path.join(folder, "metrics.json")
        if not os.path.exists(json_path): continue
        
        try:
            with open(json_path, 'r') as f: data = json.load(f)
            run_best = min(data, key=lambda x: x.get("sycophancy_rate", 1.0))
            score = run_best.get("sycophancy_rate", 1.0)
            prompt = run_best.get("prompt", "")
            
            if "sae" in name:
                if score < best_sae["score"]: best_sae = {"score": score, "prompt": prompt}
            else:
                if score < best_resid["score"]: best_resid = {"score": score, "prompt": prompt}
        except: continue
            
    return best_sae["prompt"], best_resid["prompt"]

def load_resources():
    print("Loading resources...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=HOOK_POINT, device=DEVICE)
    return model, tokenizer, sae

# =====================================================
# 2. Logic: Activations & Traces
# =====================================================

def extract_dense_vector(model, tokenizer, df):
    print("Extracting Dense Vector...")
    pos, neg = [], []
    subset = df.iloc[:100]
    for _, row in subset.iterrows():
        q = row['question']
        ans_syc = row['answer_matching_behavior']
        ans_hon = "(B)" if ans_syc == "(A)" else "(A)"
        for ans, store in [(ans_syc, pos), (ans_hon, neg)]:
            inp = tokenizer(f"{q} {ans}", return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            store.append(out.hidden_states[LAYER_IDX+1][0, -1, :])
    vec = torch.stack(pos).mean(0) - torch.stack(neg).mean(0)
    return vec / vec.norm()

@torch.no_grad()
def get_layerwise_traces(model, tokenizer, prompts, steering_vec=None):
    """
    Returns [num_layers, num_prompts, hidden_dim] tensor of residuals at last token.
    """
    all_layers_states = [] # List of lists
    
    # Register hook if needed
    handle = None
    if steering_vec is not None:
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            h += DENSE_COEF * steering_vec.to(h.device)
            return (h,) + output[1:] if isinstance(output, tuple) else h
        handle = model.model.layers[LAYER_IDX].register_forward_hook(hook)

    try:
        # We process one by one or small batch to collect ALL layers without OOM
        # Llama 3.1 8B has 33 layers (0-32). hidden_states has 33 elements.
        
        # Initialize storage
        num_layers = model.config.num_hidden_layers + 1
        storage = [[] for _ in range(num_layers)]
        
        batch_size = 8
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            out = model(**inputs, output_hidden_states=True)
            
            # Extract last token state for every layer
            for l in range(num_layers):
                # hidden_states[l]: [batch, seq, dim]
                # last token: [batch, dim]
                last_token_idxs = inputs.attention_mask.sum(dim=1) - 1
                states = out.hidden_states[l][torch.arange(len(batch)), last_token_idxs, :]
                storage[l].append(states)
                
    finally:
        if handle: handle.remove()

    # Concatenate batches per layer
    # Result: List of [Total_Prompts, Dim]
    final_states = [torch.cat(layer_batch, dim=0) for layer_batch in storage]
    
    # Stack: [Layers, Prompts, Dim]
    return torch.stack(final_states)

@torch.no_grad()
def get_sae_acts(model, tokenizer, sae, prompts):
    # Only need Layer 25 for bar plots
    resids = []
    batch_size = 16
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        out = model(**inputs, output_hidden_states=True)
        resids.append(out.hidden_states[LAYER_IDX + 1][:, -1, :])
    return torch.cat(resids), sae.encode(torch.cat(resids))

# =====================================================
# 3. Plots
# =====================================================

def plot_layerwise_mechanics(traces_base, traces_dense, traces_resid, traces_sae):
    print("Generating Figure 4: Layer-wise Mechanics...")

    # ---- FORCE CPU ONCE ----
    traces_base  = traces_base.detach().cpu()
    traces_dense = traces_dense.detach().cpu()
    traces_resid = traces_resid.detach().cpu()
    traces_sae   = traces_sae.detach().cpu()

    # 1. Shift Magnitude (L2 distance from Baseline)
    diff_dense = (traces_dense - traces_base).norm(dim=-1).mean(dim=1).numpy()
    diff_resid = (traces_resid - traces_base).norm(dim=-1).mean(dim=1).numpy()
    diff_sae   = (traces_sae   - traces_base).norm(dim=-1).mean(dim=1).numpy()

    layers = np.arange(len(diff_dense))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Magnitude of Intervention
    ax1.plot(layers, diff_dense, label=LABELS["Dense"], color=PALETTE["Dense"], linestyle="-")
    ax1.plot(layers, diff_resid, label=LABELS["Resid"], color=PALETTE["Resid"], linestyle="-")
    ax1.plot(layers, diff_sae, label=LABELS["Sae"], color=PALETTE["Sae"], linestyle="-")
    
    # Mark Injection Layer
    ax1.axvline(LAYER_IDX, color='black', linestyle=':', label="Target Layer (25)")
    
    ax1.set_title("Magnitude of Representation Shift", pad=15)
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("L2 Norm of Difference (vs Baseline)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Directional Consistency (Cosine Sim to Baseline)
    # Lower = More deviation/steering effect
    cos = torch.nn.CosineSimilarity(dim=-1)
    sim_dense = cos(traces_dense, traces_base).mean(dim=1).numpy()
    sim_resid = cos(traces_resid, traces_base).mean(dim=1).numpy()
    sim_sae = cos(traces_sae, traces_base).mean(dim=1).numpy()
    
    ax2.plot(layers, sim_dense, label=LABELS["Dense"], color=PALETTE["Dense"])
    ax2.plot(layers, sim_resid, label=LABELS["Resid"], color=PALETTE["Resid"])
    ax2.plot(layers, sim_sae, label=LABELS["Sae"], color=PALETTE["Sae"])
    ax2.axvline(LAYER_IDX, color='black', linestyle=':')
    
    ax2.set_title("Directional Deviation from Baseline", pad=15)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Cosine Similarity (with Baseline)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig4_layerwise_mechanics.png")

def plot_layer_sensitivity():
    print("Generating Figure 5: Layer Sensitivity...")
    if not os.path.exists(METRICS_CSV): return

    df = pd.read_csv(METRICS_CSV)
    subset = df[(df["Task"] == "Sycophancy") & (df["Model"] == "Llama 3.1 8B") & (df["Method"] == "Residual EPO")]
    layer_perf = subset.groupby("Layer")["Score"].min().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=layer_perf, x="Layer", y="Score", marker='o', markersize=10, linewidth=3, color=PALETTE["Resid"])
    plt.title("Layer Sensitivity (Residual EPO Performance)", pad=20)
    plt.ylabel("Sycophancy Rate (Lower is Better)")
    plt.xlabel("Layer Index")
    
    # Add Baseline
    plt.axhline(0.7248, color='grey', linestyle='--', label="Baseline (Zero-Shot)", linewidth=2)
    plt.legend()
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig5_layer_sensitivity.png")

def plot_features(a_base, a_dense, a_resid, a_sae, df, model, tokenizer, sae):
    print("Generating Figure 2: Surgical Features...")
    subset = df.iloc[:100]
    prompts_syc, prompts_hon = [], []
    for _, row in subset.iterrows():
        q = row['question']
        syc = row['answer_matching_behavior']
        hon = "(B)" if syc == "(A)" else "(A)"
        prompts_syc.append(f"{q} {syc}")
        prompts_hon.append(f"{q} {hon}")
    
    r_syc, a_syc = get_sae_acts(model, tokenizer, sae, prompts_syc)
    r_hon, a_hon = get_sae_acts(model, tokenizer, sae, prompts_hon)
    
    diff = (a_syc - a_hon).mean(0)
    top_indices = torch.topk(diff, 8).indices
    
    ch_dense = (a_dense - a_base).mean(0)[top_indices]
    ch_resid = (a_resid - a_base).mean(0)[top_indices]
    ch_sae = (a_sae - a_base).mean(0)[top_indices]
    
    valid_indices = [i for i in range(len(top_indices)) if abs(ch_dense[i]) > 0.1 or abs(ch_resid[i]) > 0.1 or abs(ch_sae[i]) > 0.1]
    
    plot_data = []
    for i in valid_indices:
        feat_id = str(top_indices[i].item())
        plot_data.append({"ID": feat_id, "Change": ch_dense[i].item(), "Method": LABELS["Dense"]})
        plot_data.append({"ID": feat_id, "Change": ch_resid[i].item(), "Method": LABELS["Resid"]})
        plot_data.append({"ID": feat_id, "Change": ch_sae[i].item(), "Method": LABELS["Sae"]})
        
    plt.figure(figsize=(14, 9))
    sns.barplot(data=pd.DataFrame(plot_data), x="ID", y="Change", hue="Method",
                palette=[PALETTE["Dense"], PALETTE["Resid"], PALETTE["Sae"]], edgecolor="black")
    
    plt.title("Impact on Features Correlated to Sycophancy", pad=20)
    plt.ylabel("Activation Change (Steered - Baseline)",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(
        "SAE Feature Index",
        fontsize=20,
        fontweight="bold"
    )
    plt.axhline(0, color="black", linewidth=2)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=3,
        frameon=True,
    )
    plt.text(0.02, 0.05, "Negative = Suppression", transform=plt.gca().transAxes, 
             fontsize=12, style='italic', bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig2_feature_impact.png")

def plot_pca(r_base, r_dense, r_resid, r_sae):
    print("Generating Figure 1: PCA Manifold...")
    all_res = torch.cat([r_base, r_dense, r_resid, r_sae])
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(all_res.cpu().numpy())
    
    n = len(r_base)
    p_base, p_dense, p_resid, p_sae = pca_res[:n], pca_res[n:2*n], pca_res[2*n:3*n], pca_res[3*n:]
    
    plt.figure(figsize=(11, 8))
    sns.kdeplot(x=p_base[:,0], y=p_base[:,1], fill=True, color=PALETTE["Base"], alpha=0.15, levels=4, thresh=0.05)
    sns.kdeplot(x=p_dense[:,0], y=p_dense[:,1], fill=True, color=PALETTE["Dense"], alpha=0.15, levels=4, thresh=0.05)
    sns.kdeplot(x=p_resid[:,0], y=p_resid[:,1], fill=True, color=PALETTE["Resid"], alpha=0.15, levels=4, thresh=0.05)
    sns.kdeplot(x=p_sae[:,0], y=p_sae[:,1], fill=True, color=PALETTE["Sae"], alpha=0.2, levels=4, thresh=0.05)
    
    s = 30
    plt.scatter(p_base[:,0], p_base[:,1], c=PALETTE["Base"], s=s, alpha=0.5, label=LABELS["Base"], edgecolors='w', linewidth=0.5)
    plt.scatter(p_dense[:,0], p_dense[:,1], c=PALETTE["Dense"], s=s, alpha=0.5, label=LABELS["Dense"], edgecolors='w', linewidth=0.5)
    plt.scatter(p_resid[:,0], p_resid[:,1], c=PALETTE["Resid"], s=s, alpha=0.5, label=LABELS["Resid"], edgecolors='w', linewidth=0.5)
    plt.scatter(p_sae[:,0], p_sae[:,1], c=PALETTE["Sae"], s=40, alpha=0.9, marker='X', label=LABELS["Sae"], edgecolors='k', linewidth=0.5)
    
    mu_base = p_base.mean(0)
    for p, col in [(p_dense, PALETTE["Dense"]), (p_resid, PALETTE["Resid"]), (p_sae, PALETTE["Sae"])]:
        mu = p.mean(0)
        plt.arrow(mu_base[0], mu_base[1], mu[0]-mu_base[0], mu[1]-mu_base[1], color=col, width=0.08, head_width=0.5, zorder=100)

    plt.title("Comparison of Steering Trajectories", pad=20)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor="black")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig1_pca_manifold.png")

def plot_sparsity(a_base, a_dense, a_resid, a_sae):
    print("Generating Figure 3: Sparsity...")
    l0_base = (a_base > 0.1).float().sum(dim=1).cpu().numpy()
    l0_dense = (a_dense > 0.1).float().sum(dim=1).cpu().numpy()
    l0_resid = (a_resid > 0.1).float().sum(dim=1).cpu().numpy()
    l0_sae = (a_sae > 0.1).float().sum(dim=1).cpu().numpy()
    
    data = []
    data.extend([{"L0": x, "Method": LABELS["Base"]} for x in l0_base])
    data.extend([{"L0": x, "Method": LABELS["Dense"]} for x in l0_dense])
    data.extend([{"L0": x, "Method": LABELS["Resid"]} for x in l0_resid])
    data.extend([{"L0": x, "Method": LABELS["Sae"]} for x in l0_sae])
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=pd.DataFrame(data), x="Method", y="L0", palette=PALETTE)
    plt.title("Sparsity Preservation (L0 Norm)", pad=20)
    plt.ylabel("Active SAE Features (Count)")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig3_sparsity.png")

def plot_logit_lens(model, tokenizer, df, prompt_sae, prompt_resid):
    print("Generating Figure 6: Logit Lens...")
    
    # 1. Identify A/B Token IDs
    # Llama 3 often uses " (A)" or just "A" depending on spacing.
    # We use the method that worked in your eval script:
    id_a = tokenizer(" (A)", add_special_tokens=False).input_ids[-1]
    id_b = tokenizer(" (B)", add_special_tokens=False).input_ids[-1]

    # 2. Select a subset of questions where Baseline is Sycophantic
    # We want to see how we FIX them.
    subset = df.sample(n=32, random_state=42)
    questions = subset['question'].tolist()
    
    def get_layer_probs(prompts, steering_vec=None):
        handle = None
        if steering_vec is not None:
             def hook(module, input, output):
                 h = output[0] if isinstance(output, tuple) else output
                 h += DENSE_COEF * steering_vec.to(h.device)
                 return (h,) + output[1:] if isinstance(output, tuple) else h
             handle = model.model.layers[LAYER_IDX].register_forward_hook(hook)

        # We will collect logits at every layer by applying the Unembed matrix
        # manually to hidden_states.
        # W_U shape: [d_model, vocab]
        W_U = model.lm_head.weight.detach() # or model.embed_out for some models
        
        layer_diffs = []
        
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            
            # Iterate layers (0 to 32)
            for layer_idx, hidden in enumerate(out.hidden_states):
                # Hidden: [batch, seq, dim] -> Last token: [batch, dim]
                # Note: We need careful masking for last token if padding exists.
                # Simplified: use last index assuming left padding or no padding issues in small batch.
                last_token_idxs = inputs.attention_mask.sum(dim=1) - 1
                last_hidden = hidden[torch.arange(len(prompts)), last_token_idxs, :]
                
                # Apply Unembed: [batch, dim] @ [dim, vocab] -> [batch, vocab]
                # Only need columns A and B
                logits_a = last_hidden @ W_U[id_a]
                logits_b = last_hidden @ W_U[id_b]
                
                # Prob Diff: Softmax approximation or just Logit Diff
                # Logit Diff is standard for Lens. Positive = Sycophancy (if A is Syc)
                # We need to know which is Syc.
                # Simplified assumption: For these prompts, let's assume A is Sycophantic
                # (You might need to align signs if your dataset mixes A/B sycophancy).
                
                # Ideally, we flip sign based on label.
                # Let's just plot Logit(A) - Logit(B) and assume A is usually sycophancy target 
                # or aggregate magnitude.
                
                diff = (logits_a - logits_b).mean().item()
                layer_diffs.append(diff)
                
        finally:
            if handle: handle.remove()
            
        return layer_diffs

    # Extract Dense Vector Again (Needed for the hook)
    dense_vec = extract_dense_vector(model, tokenizer, df)

    # Get Traces
    trace_base = get_layer_probs(questions)
    trace_dense = get_layer_probs(questions, steering_vec=dense_vec)
    trace_resid = get_layer_probs([f"{q} {prompt_resid}" for q in questions])
    trace_sae = get_layer_probs([f"{q} {prompt_sae}" for q in questions])
    
    # Plot
    plt.figure(figsize=(10, 6))
    layers = np.arange(len(trace_base))
    
    plt.plot(layers, trace_base, label=LABELS["Base"], color=PALETTE["Base"], linestyle=":", linewidth=2)
    plt.plot(layers, trace_dense, label=LABELS["Dense"], color=PALETTE["Dense"], linewidth=2)
    plt.plot(layers, trace_resid, label=LABELS["Resid"], color=PALETTE["Resid"], linewidth=3)
    plt.plot(layers, trace_sae, label=LABELS["Sae"], color=PALETTE["Sae"], linewidth=3)
    
    # Mark Injection
    plt.axvline(LAYER_IDX, color='black', linestyle='--', alpha=0.5, label="Vector Injection")
    plt.axhline(0, color='grey', linewidth=1)
    
    plt.title("Logit Lens: Decision Trajectory", pad=20)
    plt.xlabel("Layer Index")
    plt.ylabel("Logit Difference (A - B)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig6_logit_lens.png", dpi=500)
    print("Saved results/plots/fig6_logit_lens.png")

# =====================================================
# Main
# =====================================================
def main():
    model, tokenizer, sae = load_resources()
    df = pd.read_csv(DATA_PATH)
    
    prompt_sae, prompt_resid = get_best_prompts(RESULTS_DIR)
    if not prompt_sae: prompt_sae = "" 
    if not prompt_resid: prompt_resid = ""
    
    dense_vec = extract_dense_vector(model, tokenizer, df)
    
    print("Computing Traces (Layer-wise)...")
    subset = df.sample(n=32, random_state=123)
    qs = subset['question'].tolist()
    
    # 1. Layer-wise comparison (Heavy computation)
    # trace_base = get_layerwise_traces(model, tokenizer, qs)
    # trace_dense = get_layerwise_traces(model, tokenizer, qs, steering_vec=dense_vec)
    # trace_resid = get_layerwise_traces(model, tokenizer, [f"{q} {prompt_resid}" for q in qs])
    # trace_sae = get_layerwise_traces(model, tokenizer, [f"{q} {prompt_sae}" for q in qs])
    
    # plot_layerwise_mechanics(trace_base, trace_dense, trace_resid, trace_sae)
    # ---- FREE GPU MEMORY ----
    # del trace_base, trace_dense, trace_resid, trace_sae
    # torch.cuda.empty_cache()
    
    # 2. Last-Layer comparison for PCA/Features (Use more data for stability)
    print("Computing Last-Layer Activations...")
    viz_df = df.sample(n=256, random_state=42)
    q_viz = viz_df['question'].tolist()
    
    r_base, a_base = get_sae_acts(model, tokenizer, sae, q_viz)
    # For dense, we need to extract residual from steered model manually since get_sae_acts doesn't support vector arg
    # Quick fix: manually call get_residuals_and_acts helper style (but simplified here)
    # Let's just use the `get_layerwise_traces` output for layer 25 if we had enough data, 
    # but we need activations.
    
    # Let's re-implement the simple `get_residuals_and_acts` with vector support just for this block
    # (Copied from previous logic to ensure correctness)
    @torch.no_grad()
    def get_res_vec(prompts, vec=None):
        r_list = []

        handle = None
        captured = []

        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if vec is not None:
                h = h + DENSE_COEF * vec.to(h.device)
            captured.append(h[:, -1, :].detach())
            return out

        handle = model.model.layers[LAYER_IDX].register_forward_hook(hook_fn)

        try:
            for i in range(0, len(prompts), 4):   # keep batch small
                batch = prompts[i:i+4]
                inp = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(DEVICE)

                _ = model(**inp)   # ❗ NO output_hidden_states
                r_list.append(torch.cat(captured, dim=0))
                captured.clear()

        finally:
            handle.remove()

        return torch.cat(r_list, dim=0)


    r_dense = get_res_vec(q_viz, dense_vec)
    r_resid = get_res_vec([f"{q} {prompt_resid}" for q in q_viz])
    r_sae = get_res_vec([f"{q} {prompt_sae}" for q in q_viz])
    
    a_dense = sae.encode(r_dense)
    a_resid = sae.encode(r_resid)
    a_sae = sae.encode(r_sae)
    
    # plot_pca(r_base, r_dense, r_resid, r_sae)
    plot_features(a_base, a_dense, a_resid, a_sae, df, model, tokenizer, sae)
    # plot_sparsity(a_base, a_dense, a_resid, a_sae)
    
    # plot_layer_sensitivity()
    # plot_logit_lens(model, tokenizer, df, prompt_sae, prompt_resid)
    
    print(f"\nAll plots saved to {OUT_DIR}")

if __name__ == "__main__":
    main()