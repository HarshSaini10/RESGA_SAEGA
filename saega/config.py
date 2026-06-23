"""Configuration loading helpers.

All YAML configs live under ``configs/``. ``REPO_ROOT`` is resolved relative to
this file so the package works regardless of the current working directory; it
can be overridden with the ``RESGA_SAEGA_ROOT`` environment variable.
"""
import os
import yaml

REPO_ROOT = os.environ.get(
    "RESGA_SAEGA_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")


def _load(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def load_config():
    """Global defaults (model, EPO hyperparameters, fluency presets)."""
    return _load("config.yaml")


def load_personas():
    """Per-persona definitions keyed by persona name."""
    return _load("personas.yaml")


def get_persona(name):
    personas = load_personas()
    if name not in personas:
        raise KeyError(f"Unknown persona '{name}'. Available: {sorted(personas)}")
    return personas[name]


def resolve_path(path):
    """Resolve a config-relative path against the repo root."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
