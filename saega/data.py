"""Dataset loading and train/test splitting for each persona."""
import pandas as pd

from .config import resolve_path


def load_split(persona_cfg):
    """Load a persona dataset, shuffle (seed 42) and split by ``train_frac``.

    Returns ``(train_df, test_df)``.
    """
    path = resolve_path(persona_cfg["dataset"])
    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(persona_cfg["train_frac"] * len(df))
    return df.iloc[:split], df.iloc[split:]


def context_samples(train_df, persona_cfg):
    """Records used as the question context that is prepended during optimization.

    Empty for personas whose runner optimizes the prompt directly (sycophancy).
    """
    n = persona_cfg.get("context", {}).get("n_samples", 0)
    if n <= 0:
        return []
    return train_df.iloc[:n].to_dict("records")
