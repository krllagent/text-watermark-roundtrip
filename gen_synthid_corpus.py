"""Generate a corpus carrying the real SynthID text watermark.

Everything before this ran against a lexical stand-in that marked an eighth of
the words. The published scheme marks every generated token, so a stand-in
understates how hard removal is. This uses Google's reference implementation as
shipped in transformers: the same logits processor that biases sampling at
generation time, and the same detector, which needs only the text and the keys.

Both a marked and an unmarked copy of every document are produced from the same
prompt and seed, so the detector's threshold can be calibrated on text this
model actually writes rather than on an assumption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = ROOT / "results" / "synthid-corpus-v1.json"

# The reference configuration shipped with the implementation. The production
# key at any vendor is different; that does not matter here, because what is
# being measured is whether the algorithm survives a rewrite, not whether one
# particular key does.
WATERMARK_KEYS = [
    654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468,
    990, 966, 226, 324, 585, 118, 504, 421, 521, 129, 669, 732, 225, 90, 960,
]
NGRAM_LEN = 5
CONTEXT_HISTORY_SIZE = 1024

BRIEFS = [
    ("doc-01", "a public library that replaced its paper visitor book with a tablet"),
    ("doc-02", "a bicycle courier service trialling a new pickup window"),
    ("doc-03", "a school science fair judged by a rotating panel of parents"),
    ("doc-04", "a small bakery moving from phone orders to a shared spreadsheet"),
    ("doc-05", "a community garden deciding how to allocate plots for a season"),
    ("doc-06", "a museum trialling audio guides on visitors' own phones"),
    ("doc-07", "a repair cafe tracking which appliances come back twice"),
    ("doc-08", "a village bus route changing its evening timetable"),
    ("doc-09", "an office kitchen trialling a rota for the coffee machine"),
    ("doc-10", "a swimming pool changing how it books lane time"),
]

PROMPT = (
    "Write a detailed, factual report of about 600 words about {brief}. "
    "Everything is fictional but write it as a plain factual report. Include "
    "concrete numbers, dates, named people, a cost figure, at least one "
    "requirement with a deadline, and a clear recommendation at the end. "
    "Write flowing prose in paragraphs. Do not use headings, lists or bullet "
    "points."
)


def build(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    return tokenizer, model


def watermark_config() -> SynthIDTextWatermarkingConfig:
    return SynthIDTextWatermarkingConfig(
        keys=WATERMARK_KEYS,
        ngram_len=NGRAM_LEN,
        context_history_size=CONTEXT_HISTORY_SIZE,
    )


def generate(tokenizer, model, brief: str, *, marked: bool, max_new_tokens: int, seed: int):
    messages = [{"role": "user", "content": PROMPT.format(brief=brief)}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt")
    torch.manual_seed(seed)
    kwargs = dict(
        **inputs,
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
    )
    if marked:
        kwargs["watermarking_config"] = watermark_config()
    with torch.no_grad():
        out = model.generate(**kwargs)
    generated = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True), generated


def detector(tokenizer, model):
    """The mean g-value detector: the processor alone, no trained model needed."""
    config = watermark_config().to_dict()
    config.pop("skip_first_ngram_calls", None)
    config.pop("debug_mode", None)
    return SynthIDTextWatermarkLogitsProcessor(**config, device="cpu")


def mean_g(processor, tokenizer, text: str) -> float | None:
    """Mean g-value of a text: the model-free statistic the paper defines."""
    ids = tokenizer([text], return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.shape[1] <= NGRAM_LEN:
        return None
    g = processor.compute_g_values(input_ids=ids)
    mask = processor.compute_context_repetition_mask(input_ids=ids)
    valid = mask.unsqueeze(-1).expand_as(g)
    total = valid.sum().item()
    if total == 0:
        return None
    return float((g * valid).sum().item() / total)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--documents", type=int, default=len(BRIEFS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    tokenizer, model = build(args.model)
    det = detector(tokenizer, model)
    rows = []
    for index, (document_id, brief) in enumerate(BRIEFS[: args.documents]):
        started = time.monotonic()
        marked_text, _ = generate(
            tokenizer, model, brief, marked=True,
            max_new_tokens=args.max_new_tokens, seed=20260819 + index,
        )
        plain_text, _ = generate(
            tokenizer, model, brief, marked=False,
            max_new_tokens=args.max_new_tokens, seed=20260819 + index,
        )
        row = {
            "brief": brief,
            "documentId": document_id,
            "markedMeanG": mean_g(det, tokenizer, marked_text),
            "markedText": marked_text,
            "unmarkedMeanG": mean_g(det, tokenizer, plain_text),
            "unmarkedText": plain_text,
            "markedWords": len(marked_text.split()),
            "unmarkedWords": len(plain_text.split()),
        }
        rows.append(row)
        print(json.dumps({
            "documentId": document_id,
            "markedMeanG": round(row["markedMeanG"], 4) if row["markedMeanG"] else None,
            "unmarkedMeanG": round(row["unmarkedMeanG"], 4) if row["unmarkedMeanG"] else None,
            "words": row["markedWords"],
            "seconds": round(time.monotonic() - started, 1),
        }, sort_keys=True), flush=True)

    marked = [r["markedMeanG"] for r in rows if r["markedMeanG"] is not None]
    plain = [r["unmarkedMeanG"] for r in rows if r["unmarkedMeanG"] is not None]
    payload = {
        "contextHistorySize": CONTEXT_HISTORY_SIZE,
        "documents": rows,
        "keys": WATERMARK_KEYS,
        "markedMeanG": statistics.mean(marked) if marked else None,
        "model": args.model,
        "ngramLen": NGRAM_LEN,
        "schemaVersion": 1,
        "unmarkedMeanG": statistics.mean(plain) if plain else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "markedMeanG": round(payload["markedMeanG"], 4) if marked else None,
        "unmarkedMeanG": round(payload["unmarkedMeanG"], 4) if plain else None,
        "documents": len(rows),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
