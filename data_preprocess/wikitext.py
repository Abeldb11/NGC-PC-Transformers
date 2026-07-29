from datasets import load_dataset
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

# HF's split names -> the filenames this repo expects
split_map = {"train": "train.txt", "validation": "valid.txt", "test": "test.txt"}

for hf_split, filename in split_map.items():
    text = "\n".join(ds[hf_split]["text"])
    out_path = OUT_DIR / filename
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path} ({len(text):,} chars)")