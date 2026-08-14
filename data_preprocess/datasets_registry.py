"""
datasets_registry.py

Central registry for dataset selection in the NGC-PC-Transformers pipeline.

Given the `dataset` value set in config.py, this module resolves it to a
concrete (data_dir, output_dir) pair on disk, downloading or migrating the
raw text the first time a dataset is used. Every dataset gets its own
subfolder under data_preprocess/data/ and data_preprocess/outputs/, so
switching datasets is a one-line change in config.py with no manual file
handling and no redundant re-downloading or re-tokenizing on repeat runs.

Supported values for config.dataset (case/hyphen/space/underscore-insensitive):
    "tinyshakespeare"   -> Tiny Shakespeare (bundled with the repo) has 200K words
    "wikitext2"          -> WikiText-2-raw-v1  (auto-downloaded via HF datasets) has 2,550K words
    "wikitext103"        -> WikiText-103-raw-v1 (auto-downloaded via HF datasets) 103,690K words
    "ptb"                -> Penn Treebank (auto-downloaded via HF datasets) 1000K words
    "rottentomatoes"      -> Rotten Tomatoes movie review sentences (auto-downloaded via HF datasets) 230K words

Usage:
    from data_preprocess.datasets_registry import prepare_dataset
    data_dir, output_dir = prepare_dataset(config.dataset)
"""

import re
import urllib.request
from pathlib import Path
from typing import Tuple

DIR = Path(__file__).parent
DATA_ROOT = DIR / "data"
OUTPUT_ROOT = DIR / "outputs"

SPLIT_NAMES = ("train", "valid", "test")

# Canonical dataset name -> list of accepted aliases (normalized internally)
_ALIASES = {
    "tinyshakespeare": ["tinyshakespeare", "tinyshakespear", "shakespeare", "tiny_shakespeare"],
    "wikitext2": ["wikitext2", "wikitext-2", "wt2"],
    "wikitext103": ["wikitext103", "wikitext-103", "wt103"],
    "ptb": ["ptb", "penntreebank", "penn-treebank", "penn_treebank", "ptbtextonly", "ptb_text_only"],
    "rottentomatoes": ["rottentomatoes", "rotten-tomatoes", "rotten_tomatoes", "rt", "moviereviews"],
}

# Hugging Face `Salesforce/wikitext` config name for each wikitext variant


_HF_DATASETS = {
    "wikitext2": {
        "path": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "text_field": "text",
        "splits": {"train": "train", "validation": "valid", "test": "test"},
    },
    "wikitext103": {
        "path": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "text_field": "text",
        "splits": {"train": "train", "validation": "valid", "test": "test"},
    },
    "ptb": {
        "path": "ptb-text-only/ptb_text_only",
        "config": None,
        "text_field": "sentence",
        "splits": {"train": "train", "validation": "valid", "test": "test"},
    },
    "rottentomatoes": {
        "path": "cornell-movie-review-data/rotten_tomatoes",
        "config": None,
        "text_field": "text",
        "splits": {"train": "train", "validation": "valid", "test": "test"},
    },
}

# Fallback source if no local Tiny Shakespeare files are found at all
_TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _normalize(name: str) -> str:
    """Map a user-typed dataset name to its canonical registry key."""
    key = re.sub(r"[\s\-_]", "", name.lower())
    for canonical, aliases in _ALIASES.items():
        if key in (re.sub(r"[\s\-_]", "", alias) for alias in aliases):
            return canonical
    raise ValueError(
        f"Unknown dataset '{name}'. Valid options: {sorted(_ALIASES.keys())}"
    )


def _splits_present(data_dir: Path) -> bool:
    """True only if train/valid/test.txt all exist and are non-empty."""
    return all((data_dir / f"{split}.txt").exists() and (data_dir / f"{split}.txt").stat().st_size > 0
               for split in SPLIT_NAMES)


def _cleanup_partial(data_dir: Path) -> None:
    """Remove any partially-written split files after a failed download."""
    for split in SPLIT_NAMES:
        f = data_dir / f"{split}.txt"
        if f.exists():
            f.unlink()


def _migrate_legacy_tinyshakespeare(data_dir: Path) -> bool:
    """
    One-time move: picks up train/valid/test.txt from the old flat layout
    (data_preprocess/data/*.txt) if they're still sitting there, and relocates
    them into data/tinyshakespeare/. Purely local, no network involved.
    Returns True if a migration happened.
    """
    legacy_paths = {split: DATA_ROOT / f"{split}.txt" for split in SPLIT_NAMES}
    if not all(p.exists() for p in legacy_paths.values()):
        return False

    print("[datasets_registry] Found legacy flat-layout Shakespeare files — migrating...")
    for split, old_path in legacy_paths.items():
        old_path.rename(data_dir / f"{split}.txt")
    return True


def _download_tinyshakespeare(data_dir: Path) -> None:
    """Fallback only: fetch Tiny Shakespeare from GitHub and split 90/5/5."""
    print("[datasets_registry] No local Shakespeare files found — downloading fallback copy...")
    try:
        with urllib.request.urlopen(_TINYSHAKESPEARE_URL) as response:
            text = response.read().decode("utf-8")

        n = len(text)
        train_end, valid_end = int(n * 0.90), int(n * 0.95)
        (data_dir / "train.txt").write_text(text[:train_end], encoding="utf-8")
        (data_dir / "valid.txt").write_text(text[train_end:valid_end], encoding="utf-8")
        (data_dir / "test.txt").write_text(text[valid_end:], encoding="utf-8")
    except Exception:
        _cleanup_partial(data_dir)
        raise


def _download_hf_dataset(canonical_name: str, data_dir: Path) -> None:
    """Download and split any dataset registered in _HF_DATASETS via the Hugging Face `datasets` library."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            f"The 'datasets' package is required to download '{canonical_name}'. "
            "Install it with: pip install datasets"
        ) from e

    spec = _HF_DATASETS[canonical_name]
    config_note = f", config='{spec['config']}'" if spec["config"] else ""
    print(f"[datasets_registry] Downloading '{canonical_name}' from Hugging Face "
          f"({spec['path']}{config_note})...")
    try:
        ds = load_dataset(spec["path"], spec["config"]) if spec["config"] else load_dataset(spec["path"])
        for hf_split, local_split in spec["splits"].items():
            text = "\n".join(ds[hf_split][spec["text_field"]])
            filename = f"{local_split}.txt"
            (data_dir / filename).write_text(text, encoding="utf-8")
            print(f"  wrote {filename} ({len(text):,} chars)")
    except Exception:
        _cleanup_partial(data_dir)
        raise

def prepare_dataset(name: str) -> Tuple[Path, Path]:
    """
    Resolve config.dataset to concrete local paths, downloading or migrating
    raw text on first use. Subsequent calls with the same dataset are
    effectively free (cache hit).

    Args:
        name: dataset identifier, e.g. "tinyshakespeare", "wikitext2", "wikitext103"
              (case/hyphen/space/underscore-insensitive).

    Returns:
        (data_dir, output_dir): data_dir holds train/valid/test.txt;
        output_dir is where the tokenizer and tokenized .npy files for
        this dataset should be saved/loaded from.
    """
    canonical = _normalize(name)
    data_dir = DATA_ROOT / canonical
    output_dir = OUTPUT_ROOT / canonical
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _splits_present(data_dir):
        print(f"[datasets_registry] Using cached '{canonical}' data at {data_dir}")
        return data_dir, output_dir


    if canonical == "tinyshakespeare":
        if not _migrate_legacy_tinyshakespeare(data_dir):
            _download_tinyshakespeare(data_dir)
    else:
        _download_hf_dataset(canonical, data_dir)   # was: _download_wikitext(canonical, data_dir)

    if not _splits_present(data_dir):
        raise RuntimeError(
            f"Failed to prepare dataset '{canonical}': train/valid/test.txt "
            f"were not all created in {data_dir}"
        )

    return data_dir, output_dir


def list_available_datasets() -> list:
    """Return the canonical dataset names this registry currently supports."""
    return sorted(_ALIASES.keys())