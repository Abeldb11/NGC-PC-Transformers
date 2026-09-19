"""
generation_eval.py

Objective, self-relative generation-quality measurement.

This does NOT compare your model against the backprop baseline. It compares
each generation *run* (e.g. "before bos fix", "after bos fix", "after
settled-inference fix", or just successive checkpoints) against your own
model's real held-out text, and tells you whether generation moved closer to
or further from that reference -- i.e. improvement or "deprovment".

No external dependencies beyond the Python standard library, and no
dependency on JAX/ngclearn/your model code -- it only ever looks at text.

────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────
1. Save each generation run as a folder of .txt files, one generated sample
   per file:

       samples/baseline/sample_00.txt
       samples/baseline/sample_01.txt
       ...
       samples/after_bos_fix/sample_00.txt
       ...

   (see `save_samples_for_run()` below for a one-line helper you can call
   directly from generation.py to produce this layout.)

2. Run:

    python generation_eval.py \
        --train_text  outputs/tinyshakespeare/train.txt \
        --valid_text  outputs/tinyshakespeare/valid.txt \
        --run baseline=samples/baseline \
        --run after_bos_fix=samples/after_bos_fix \
        --run after_settled_fix=samples/after_settled_fix

   Runs are compared in the order given: each is scored against the
   reference (your real validation text) AND against the previous run, so
   you get both "are we close to real text" and "did this change help".

3. Every run's metrics + composite score get appended to
   generation_eval_history.jsonl (override with --history_file), so you can
   plot the trend across many sessions later without re-running old samples.
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import gzip
import json
import math
import re
import statistics
import string
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

START = "\x00"  # padding symbol for the char n-gram LM


# ══════════════════════════════════════════════════════════════════════════
# Reference language model: a plain add-k character n-gram model trained on
# YOUR OWN train split. Used only as a fixed, cheap yardstick for "does this
# look like real text from this corpus" -- not a claim of SOTA LM quality.
# ══════════════════════════════════════════════════════════════════════════
class CharNgramLM:
    def __init__(self, n=5, k=0.5):
        self.n = n
        self.k = k
        self.context_counts = defaultdict(Counter)
        self.vocab = set()
        self.vocab_size = 1

    def fit(self, text):
        n = self.n
        padded = START * (n - 1) + text
        for i in range(len(text)):
            context = padded[i:i + n - 1]
            nxt = padded[i + n - 1]
            self.context_counts[context][nxt] += 1
            self.vocab.add(nxt)
        self.vocab_size = max(len(self.vocab), 1)

    def score(self, text):
        """Returns (avg_log_prob_per_char, perplexity)."""
        if not text:
            return float("nan"), float("nan")
        n = self.n
        padded = START * (n - 1) + text
        total_logprob = 0.0
        for i in range(len(text)):
            context = padded[i:i + n - 1]
            nxt = padded[i + n - 1]
            counts = self.context_counts.get(context, Counter())
            total = sum(counts.values())
            p = (counts.get(nxt, 0) + self.k) / (total + self.k * self.vocab_size)
            total_logprob += math.log(max(p, 1e-12))
        avg_logprob = total_logprob / len(text)
        ppl = math.exp(-avg_logprob)
        return avg_logprob, ppl


# ══════════════════════════════════════════════════════════════════════════
# Individual metrics
# ══════════════════════════════════════════════════════════════════════════
def word_tokens(text):
    return text.split()


def distinct_n(texts, n):
    """Unique n-grams / total n-grams, pooled across all texts in the run."""
    all_ngrams = Counter()
    total = 0
    for text in texts:
        toks = word_tokens(text)
        if len(toks) < n:
            continue
        grams = list(zip(*[toks[i:] for i in range(n)]))
        all_ngrams.update(grams)
        total += len(grams)
    return (len(all_ngrams) / total) if total else 0.0


def repetition_rate(text, window=8):
    """Fraction of tokens that repeat a token seen in the last `window` tokens."""
    toks = word_tokens(text)
    if len(toks) <= window:
        return 0.0
    hits = sum(1 for i in range(window, len(toks)) if toks[i] in toks[i - window:i])
    return hits / (len(toks) - window)


def compression_ratio(text):
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(gzip.compress(raw)) / len(raw)


def vocab_validity(text, vocab):
    """Fraction of whitespace tokens (stripped of punctuation) present in a
    reference vocab built from the training corpus. Catches garbled/gibberish
    output. Not meaningful for character-level generation shorter than a word."""
    toks = word_tokens(text)
    if not toks:
        return 0.0
    valid = sum(1 for t in toks if t.strip(string.punctuation).lower() in vocab)
    return valid / len(toks)


def _ngram_counts(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def _bleu(hyp_text, ref_texts, max_n=4):
    hyp = word_tokens(hyp_text)
    if not hyp:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngram_counts(hyp, n)
        if not hyp_ngrams:
            precisions.append(1e-9)
            continue
        max_ref_counts = Counter()
        for ref_text in ref_texts:
            ref_ngrams = _ngram_counts(word_tokens(ref_text), n)
            for g, c in ref_ngrams.items():
                if c > max_ref_counts[g]:
                    max_ref_counts[g] = c
        clipped = sum(min(c, max_ref_counts.get(g, 0)) for g, c in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        precisions.append(max(clipped / total, 1e-9))
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    ref_lens = [len(word_tokens(r)) for r in ref_texts] or [len(hyp)]
    closest = min(ref_lens, key=lambda rl: (abs(rl - len(hyp)), rl))
    bp = 1.0 if len(hyp) > closest else math.exp(1 - closest / max(len(hyp), 1))
    return bp * geo_mean


def self_bleu(texts, max_n=4, cap=40):
    """Average pairwise BLEU of each sample against all others in the same
    run. High self-BLEU / low distinct-n together = degenerate repetition
    across samples, not just within one. Capped at `cap` samples for speed."""
    texts = texts[:cap]
    if len(texts) < 2:
        return float("nan")
    scores = [_bleu(texts[i], texts[:i] + texts[i + 1:], max_n) for i in range(len(texts))]
    return sum(scores) / len(scores)


SPEAKER_TAG_RE = re.compile(r"^[A-Z][A-Z ]{1,30}:\s*$", re.MULTILINE)


def speaker_tag_rate(text):
    """Corpus-specific bonus metric for scripts/plays (e.g. tinyshakespeare):
    fraction of non-empty lines that look like a 'CHARACTER NAME:' cue.
    Harmless (near 0 for non-dramatic corpora) if this doesn't apply to you."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    tags = len(SPEAKER_TAG_RE.findall(text))
    return tags / len(lines)


# ══════════════════════════════════════════════════════════════════════════
# Per-run aggregation
# ══════════════════════════════════════════════════════════════════════════
PER_SAMPLE_METRICS = ("lm_ppl", "repetition_rate", "compression_ratio",
                      "vocab_validity", "speaker_tag_rate", "length_words")

# direction of "better" for each aggregate metric:
#   "closer" = closer to the reference value is better
#   "higher" = raw higher is always better
#   "lower"  = raw lower is always better
METRIC_DIRECTION = {
    "lm_ppl_mean": "closer",
    "repetition_rate_mean": "lower",
    "compression_ratio_mean": "closer",
    "vocab_validity_mean": "higher",
    "speaker_tag_rate_mean": "closer",
    "length_words_mean": "closer",
    "distinct_1": "higher",
    "distinct_2": "higher",
    "distinct_3": "higher",
    "self_bleu": "closer",
}


def evaluate_texts(texts, lm, train_vocab):
    """texts: list[str], one generated (or reference) sample each."""
    if not texts:
        raise ValueError("evaluate_texts() got an empty list of samples.")
    per_sample = []
    for t in texts:
        _, ppl = lm.score(t)
        per_sample.append({
            "lm_ppl": ppl,
            "repetition_rate": repetition_rate(t),
            "compression_ratio": compression_ratio(t),
            "vocab_validity": vocab_validity(t, train_vocab),
            "speaker_tag_rate": speaker_tag_rate(t),
            "length_words": len(word_tokens(t)),
        })
    agg = {}
    for key in PER_SAMPLE_METRICS:
        vals = [s[key] for s in per_sample if not math.isnan(s[key])]
        agg[f"{key}_mean"] = statistics.mean(vals) if vals else float("nan")
        agg[f"{key}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    agg["distinct_1"] = distinct_n(texts, 1)
    agg["distinct_2"] = distinct_n(texts, 2)
    agg["distinct_3"] = distinct_n(texts, 3)
    agg["self_bleu"] = self_bleu(texts)
    agg["n_samples"] = len(texts)
    return agg


def composite_score(agg, reference):
    """One headline number in [0, ~1]: 1 - mean relative distance-to-reference
    across metrics (direction-aware). Higher = closer to real text overall.
    This is a diagnostic summary, not a rigorous statistic -- always look at
    the per-metric table too."""
    dists = []
    for key, direction in METRIC_DIRECTION.items():
        if key not in agg or key not in reference:
            continue
        v, r = agg[key], reference[key]
        if v is None or r is None or (isinstance(v, float) and math.isnan(v)):
            continue
        scale = abs(r) + 1e-6
        if direction == "closer":
            d = abs(v - r) / scale
        elif direction == "higher":
            d = max(0.0, r - v) / scale
        else:  # "lower"
            d = max(0.0, v - r) / scale
        dists.append(min(d, 3.0))  # cap outlier influence
    if not dists:
        return float("nan")
    return max(0.0, 1.0 - statistics.mean(dists))


# ══════════════════════════════════════════════════════════════════════════
# Loading text
# ══════════════════════════════════════════════════════════════════════════
def load_run_samples(spec):
    """spec: a directory of .txt files (one sample per file), OR a single
    .txt file containing multiple samples separated by a line of 5+ '=' or
    '-' characters."""
    path = Path(spec)
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        if not files:
            raise ValueError(f"No .txt files found in {path}")
        return [f.read_text(encoding="utf-8").strip() for f in files]
    if path.is_file():
        raw = path.read_text(encoding="utf-8")
        parts = re.split(r"\n[=\-]{5,}\n", raw)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            raise ValueError(f"No samples found in {path}")
        return parts
    raise FileNotFoundError(spec)


def make_reference_chunks(valid_text, target_words, n_chunks=40):
    """Slice the real held-out text into chunks roughly the same length as
    your generations, so the reference is measured on a fair footing."""
    words = valid_text.split()
    if len(words) < target_words * 2:
        target_words = max(10, len(words) // (n_chunks + 1))
    chunks = []
    step = max(1, len(words) // n_chunks)
    for i in range(0, len(words) - target_words, step):
        chunks.append(" ".join(words[i:i + target_words]))
        if len(chunks) >= n_chunks:
            break
    if not chunks:
        chunks = [valid_text]
    return chunks


def build_vocab(train_text):
    toks = word_tokens(train_text)
    return {t.strip(string.punctuation).lower() for t in toks}


def save_samples_for_run(run_dir, samples):
    """Convenience helper -- call this from generation.py:

        from generation_eval import save_samples_for_run
        outs = [generate_text(model, tokenizer, prompt=p, ...) for p in prompts]
        save_samples_for_run("samples/after_bos_fix", outs)
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(samples):
        (run_dir / f"sample_{i:03d}.txt").write_text(s, encoding="utf-8")
    return run_dir


# ══════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════
def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  n/a "
    return f"{v:.{nd}f}"


def verdict(v, prev, direction, tol=0.02):
    """Compare v to prev (previous run OR reference) for one metric."""
    if v is None or prev is None or (isinstance(v, float) and math.isnan(v)) \
            or (isinstance(prev, float) and math.isnan(prev)):
        return "  n/a  "
    scale = abs(prev) + 1e-6
    rel = (v - prev) / scale
    if direction == "higher":
        signal = rel
    elif direction == "lower":
        signal = -rel
    else:  # closer: shrinking |distance| is good
        signal = abs(prev) - abs(v)  # placeholder, replaced by caller for "closer"
    if abs(rel) < tol:
        return " FLAT  "
    return "IMPROVE" if signal > 0 else "REGRESS"


def print_report(reference, runs):
    """runs: list of (name, agg_dict), in chronological order."""
    print("\n" + "═" * 78)
    print("GENERATION QUALITY -- self-relative report (reference = your real held-out text)")
    print("═" * 78)

    ref_score = composite_score(reference, reference)
    print(f"\nReference (real text) composite score: {ref_score:.3f}  (defines '1.0 / natural')")

    header = f"{'metric':<24}{'reference':>12}"
    for name, _ in runs:
        header += f"{name:>16}"
    print("\n" + header)
    print("-" * len(header))

    all_keys = list(METRIC_DIRECTION.keys())
    for key in all_keys:
        row = f"{key:<24}{fmt(reference.get(key)):>12}"
        for _, agg in runs:
            row += f"{fmt(agg.get(key)):>16}"
        print(row)

    print("-" * len(header))
    comp_row = f"{'composite (vs reference)':<24}{'1.0000':>12}"
    for _, agg in runs:
        comp_row += f"{composite_score(agg, reference):>16.4f}"
    print(comp_row)

    # Trend: each run vs the previous run (or vs reference for the first run)
    print("\n" + "─" * 78)
    print("TREND (each run vs. the one before it; first run vs. reference)")
    print("─" * 78)
    prev_agg = reference
    prev_name = "reference"
    for name, agg in runs:
        prev_score = composite_score(prev_agg, reference)
        this_score = composite_score(agg, reference)
        delta = this_score - prev_score
        tag = "IMPROVED" if delta > 0.005 else ("DEGRADED" if delta < -0.005 else "~flat")
        arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
        print(f"  {prev_name:>16} {arrow} {name:<20} composite {prev_score:.3f} -> {this_score:.3f} "
              f"({delta:+.3f})  [{tag}]")
        prev_agg, prev_name = agg, name
    print("═" * 78 + "\n")


def append_history(history_file, reference, runs):
    ts = datetime.now(timezone.utc).isoformat()
    with open(history_file, "a", encoding="utf-8") as f:
        for name, agg in runs:
            record = {
                "timestamp": ts,
                "run_name": name,
                "composite_score": composite_score(agg, reference),
                "metrics": agg,
            }
            f.write(json.dumps(record) + "\n")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_text", required=True, help="Path to raw train.txt (trains the reference n-gram LM).")
    ap.add_argument("--valid_text", required=True, help="Path to raw valid.txt (builds the reference band).")
    ap.add_argument("--run", action="append", required=True, metavar="NAME=PATH",
                     help="A generation run: NAME=path/to/dir_of_txt_files (repeatable, in chronological order).")
    ap.add_argument("--lm_order", type=int, default=5, help="Char n-gram order for the reference LM (default 5).")
    ap.add_argument("--history_file", default="generation_eval_history.jsonl")
    ap.add_argument("--no_log", action="store_true", help="Don't append to the history file.")
    args = ap.parse_args()

    train_text = Path(args.train_text).read_text(encoding="utf-8")
    valid_text = Path(args.valid_text).read_text(encoding="utf-8")

    print("Training reference n-gram LM on train_text ...")
    lm = CharNgramLM(n=args.lm_order)
    lm.fit(train_text)
    vocab = build_vocab(train_text)

    runs = []
    for spec in args.run:
        if "=" not in spec:
            print(f"ERROR: --run must be NAME=PATH, got: {spec}", file=sys.stderr)
            sys.exit(1)
        name, path = spec.split("=", 1)
        runs.append((name, load_run_samples(path)))

    avg_len = int(statistics.mean(len(word_tokens(t)) for _, texts in runs for t in texts))
    ref_texts = make_reference_chunks(valid_text, target_words=max(avg_len, 20))

    print(f"Scoring reference band ({len(ref_texts)} real chunks, ~{avg_len} words each) ...")
    reference = evaluate_texts(ref_texts, lm, vocab)

    scored_runs = []
    for name, texts in runs:
        print(f"Scoring run '{name}' ({len(texts)} samples) ...")
        scored_runs.append((name, evaluate_texts(texts, lm, vocab)))

    print_report(reference, scored_runs)

    if not args.no_log:
        append_history(args.history_file, reference, scored_runs)
        print(f"Appended results to {args.history_file}")


if __name__ == "__main__":
    main()