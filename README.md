# Text Watermark Round-Trip

A transparent CPU-only experiment for one narrow question: what happens to a
keyed lexical signal after synonym edits, translation loops, or a full
paraphrase?

This repository does not detect a Claude, Gemini, or SynthID watermark. The
marker and detector here are a deliberately simple teaching model. Their code,
keyed choices, corpus, prompts, and results are meant to be inspectable from end
to end.

## Current status

The toy marker, detector, protected-text contract, frozen 20-document corpus,
manual context review, and corpus-level statistical controls are complete. The
controls make no model calls and need no GPU. The transformation matrix has not
run yet, so this repository does not claim a winning removal method.

## How the toy marker works

The lexicon contains two-member synonym classes such as `big / large`. At each
eligible word, two domain-separated HMAC calculations make independent keyed
choices.

1. `activate-v1` decides whether the position is active at the configured
   density.
2. `favor-v1` chooses the preferred member of the pair.

The encoder uses the preferred word at active positions. The detector repeats
the calculation with the scoring key and counts preferred words only among the
active positions selected by that key.

`densityBps=1000` means ten percent of eligible synonym positions. It does not
mean ten percent of every word, and it says nothing about the density of a
production watermark. Every result reports eligible positions, active
positions, actual changes, and active coverage across the full text.

The context hash uses the previous four normalized tokens. Both members of a
synonym class normalize to the same class ID. Repeated identical fingerprints
receive a zero-based occurrence rank, which keeps the pseudo-random trials
distinct without adding a fragile global word index.

## Detection rule

The main score pools counts across a fixed corpus while preserving each
document ID in the HMAC input. For `n` active positions and `hits` preferred
words, the display score is

```text
z = (2 * hits - n) / sqrt(n)
```

The decision uses the exact one-sided binomial tail under the toy random-key
null.

```text
P[Binomial(n, 0.5) >= hits]
```

Fewer than 20 active positions returns `insufficient_evidence`; `p` and `z`
remain null. Otherwise, `p <= 0.01` returns `detected`. The exact numerator and
denominator are authoritative; decimal `p` and `z` are display values.

## Run the tests

Python 3.11 or newer is enough.

```bash
python3 -m unittest discover -s tests -v
```

The suite covers golden HMAC vectors, exact binomial boundaries, protected
spans, density monotonicity, idempotent marking, corpus pooling, and wrong-key
controls.

Run the deterministic synthetic preflight with 1,000 wrong keys:

```bash
python3 run_controls.py \
  --key-hex 00112233445566778899aabbccddeeff102132435465768798a9babbdcddedef \
  --density-bps 1000 \
  --wrong-key-count 1000 \
  --output results/synthetic-preflight-v1.json
```

Add `--check` to regenerate the same run in memory and fail if the checked-in
artifact differs byte for byte.

On the frozen synthetic fixture, the true key recovered all 133 active
positions (`z=11.53`), while the unmarked corpus was not detected. Nine of
1,000 wrong keys crossed the fixed one-percent threshold. All 1,000 wrong-key
scores had enough evidence. These are detector plumbing checks, not results for
the later transformation article.

The frozen article corpus has its own fail-closed checks.

```bash
python3 run_corpus_controls.py
python3 run_corpus_controls.py --check
python3 run_experiment.py --dry-run
```

The corpus contains 20 documents, 12,777 words, and 834 synonym occurrences
whose local contexts were reviewed before the watermark key was applied. The
current canonical control artifact is
`results/corpus-controls-v1.json` (SHA-256
`03e4efa807a05ec51bbf674c2c3ed7eca3e5f17415cbcf452e3135b4154ab0b0`).

| Configured density over eligible positions | Active positions | Active share of all words | Marked true-key z | Unmarked true-key z | Wrong-key detections |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5% | 45 | 0.352% | 6.71 | 0.15 | 12 / 1,000 |
| 10% | 97 | 0.759% | 9.85 | -0.10 | 9 / 1,000 |
| 20% | 176 | 1.377% | 13.27 | 0.15 | 9 / 1,000 |

At all three densities, the marked corpus was detected, the unmarked corpus was
not detected, and every wrong-key score had enough evidence. At the main 10%
setting, only 0.384% of all words actually changed during encoding. Density is
therefore not interchangeable with the percentage of all words visibly edited.

## Use the CLI

The CLI reads a UTF-8 file and prints canonical JSON. This sample key is public
and only demonstrates the toy scheme.

```bash
python3 watermark_toy.py \
  encode \
  --key-hex 00112233445566778899aabbccddeeff102132435465768798a9babbdcddedef \
  --document-id example-1 \
  --density-bps 1000 \
  --input example.txt
```

Score a marked or transformed version with the same document ID:

```bash
python3 watermark_toy.py \
  detect \
  --key-hex 00112233445566778899aabbccddeeff102132435465768798a9babbdcddedef \
  --document-id example-1 \
  --density-bps 1000 \
  --input marked.txt
```

This command returns a single-document diagnostic. The experiment's primary
decision pools counts across all frozen corpus documents with `score_corpus`;
it does not average per-document decisions.

Document ID is part of this toy scheme. The detector cannot score an arbitrary
copied passage without the ID used by the encoder. That is a known limitation,
not a property claimed for production systems.

## Evidence boundary

The experiment can report that a named transformation changed this published
toy score from A to B. It can also show how much wording moved, whether claims
survived, and what the calls cost. It cannot prove that the same transformation
removes a secret production watermark.

The synonym pairs are not interchangeable in every English sentence. Corpus
authors must use them only where both variants preserve the intended meaning,
then manually inspect every marked document before transformation calls.

## Planned experiment

The fixed comparison has four methods.

- limited synonym edits;
- semantic round-trip through German;
- semantic round-trip through Chinese;
- a full semantic paraphrase.

German and Chinese are parameters, not predetermined winners. The final table
will put signal, factual fidelity, voice change, latency, and actual cost next
to each other. A web demo is a later decision based on those results.

## License

MIT
