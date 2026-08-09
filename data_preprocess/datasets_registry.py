"""
datasets_registry.py
Central registry for dataset selection in the NGC-PC-Transformers pipeline.

Usage
    from data_preprocess.datasets_registry import prepare_dataset
    data_dir, output_dir = prepare_dataset(config.dataset)
"""

import re
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
}

# Hugging Face `Salesforce/wikitext` config name for each wikitext variant
_HF_CONFIG = {
    "wikitext2": "wikitext-2-raw-v1",
    "wikitext103": "wikitext-103-raw-v1",
}


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


def _resolve_tinyshakespeare(data_dir: Path) -> None:
  
    legacy_paths = {split: DATA_ROOT / f"{split}.txt" for split in SPLIT_NAMES}
    if not all(p.exists() for p in legacy_paths.values()):
        raise FileNotFoundError(
            "Tiny Shakespeare data not found. Expected train/valid/test.txt either "
            f"in {data_dir} or in the legacy location {DATA_ROOT}. "
            "This dataset is sourced from the repo only — it is not downloaded."
        )

    print("[datasets_registry] Found legacy flat-layout Shakespeare files — migrating...")
    for split, old_path in legacy_paths.items():
        old_path.rename(data_dir / f"{split}.txt")


def _download_wikitext(canonical_name: str, data_dir: Path) -> None:
    """Download and split a WikiText variant via the Hugging Face datasets library."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to download WikiText. "
            "Install it with: pip install datasets"
        ) from e

    print(f"[datasets_registry] Downloading {canonical_name} from Hugging Face "
          f"(config='{_HF_CONFIG[canonical_name]}')...")
    try:
        ds = load_dataset("Salesforce/wikitext", _HF_CONFIG[canonical_name])
        split_map = {"train": "train.txt", "validation": "valid.txt", "test": "test.txt"}
        for hf_split, filename in split_map.items():
            text = "\n".join(ds[hf_split]["text"])
            (data_dir / filename).write_text(text, encoding="utf-8")
            print(f"  wrote {filename} ({len(text):,} chars)")
    except Exception:
        _cleanup_partial(data_dir)
        raise


def prepare_dataset(name: str) -> Tuple[Path, Path]:
    """
    Resolve config.dataset to concrete local paths. Tiny Shakespeare is
    located/migrated locally; WikiText-2/103 are downloaded on first use.
    Subsequent calls with the same dataset are free.

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
        _resolve_tinyshakespeare(data_dir)
    else:
        _download_wikitext(canonical, data_dir)

    if not _splits_present(data_dir):
        raise RuntimeError(
            f"Failed to prepare dataset '{canonical}': train/valid/test.txt "
            f"were not all created in {data_dir}"
        )

    return data_dir, output_dir


def list_available_datasets() -> list:
    
    return sorted(_ALIASES.keys())