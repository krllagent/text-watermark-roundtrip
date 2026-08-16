# Text Watermark Round-Trip

A transparent CPU-only experiment for one narrow question: what happens to a
keyed lexical signal after synonym edits, translation loops, or a full
paraphrase?

This repository does not detect a Claude, Gemini, or SynthID watermark. The
marker and detector here are a deliberately simple teaching model. Their code,
keyed choices, corpus, prompts, and results are meant to be inspectable from end
to end.

## Current status

The CPU-only detector controls, 20-document transformation matrix, 10,000-draw
paired bootstrap, blinded 80-pair semantic audit, and preselected 12-pair close
reading are complete. The checked-in artifacts contain every transformed text,
provider response, score, cost, and review finding. No GPU or Colab run was
used.

The main result is a tradeoff, not a universal remover. Limited synonym edits
preserved meaning but left the toy signal detected. German and Chinese
round-trips removed the toy detection but failed the blinded fidelity rule in
13/20 and 12/20 documents. Full paraphrase also removed the detection, cost
less than either translation loop, and failed fidelity in 6/20 documents. That
is the best balance in this experiment, but a 30% document failure rate is not
safe enough for unattended rewriting.

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

## Completed transformation result

The fixed comparison used one transformation model for all methods:
`qwen/qwen3.5-9b`, pinned through OpenRouter to the DeepInfra BF16 endpoint with
fallback disabled. The frozen price was $0.10 per million input tokens and
$0.15 per million output tokens. Keeping the model fixed isolates the method;
translation still costs more because each document requires two calls.

The main marker density was 10% of the 834 eligible synonym positions. This
activated 97 positions, or 0.759% of all 12,777 words. The baseline had 97/97
preferred hits (`z=9.85`, `p=6.31e-30`).

| Method | Toy detector | Original active positions retained | Mean word distance | Blinded fidelity failures | Cost, 20 docs | Cost / 1,000 docs | Median latency / doc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No transform | detected (`z=9.85`) | 100.0% | 0.0% | not audited | $0 | $0 | 0 s |
| Limited synonyms | detected (`z=6.19`) | 83.5% | 5.7% | 0 / 20 | $0.00409255 | $0.20462750 | 22.0 s |
| German round-trip | not detected (`z=1.48`) | 37.1% | 24.0% | 13 / 20 | $0.00969590 | $0.48479500 | 54.5 s |
| Chinese round-trip | not detected (`z=-0.45`) | 19.6% | 42.3% | 12 / 20 | $0.00800110 | $0.40005500 | 42.8 s |
| Full paraphrase | not detected (`z=-0.11`) | 24.7% | 29.2% | 6 / 20 | $0.00417150 | $0.20857500 | 23.9 s |

The active-position survival changes versus no transform were nonzero across
all 10,000 paired bootstrap draws. The 95% intervals were -30.6 to -4.7
percentage points for synonyms, -73.4 to -52.9 for German, -87.9 to -73.3 for
Chinese, and -84.8 to -61.6 for paraphrase. These intervals describe this
frozen corpus, not production watermark robustness.

The blinded audit used `google/gemini-3.7-flash`, pinned to Google Vertex global
with low reasoning, ZDR, and no fallback. It reviewed all 80 source-candidate
pairs without method, score, cost, or latency labels and cost $0.076574625.
A separate close read used the 12 pairs selected before transformation. It
agreed with the structured judge on 9/12 verdicts and was stricter on three
translation outputs. All three synonym samples and all three paraphrase samples
passed that small manual sample; all three German and all three Chinese samples
failed. The sample is a qualitative cross-check, not a replacement for the
full 80-pair audit.

The transformation calls cost $0.02596105. Transformation plus independent
audit cost $0.102535675 in provider charges. Two rejected route canaries were
confirmed uncharged and are recorded in
`results/semantic-audit-route-canary-failure-v1.json`.

## Reproduce the experiment

The deterministic controls and checked-in result validation require no API
key:

```bash
python3 run_corpus_controls.py --check
python3 run_experiment.py --dry-run
python3 run_semantic_audit.py --dry-run
python3 -m unittest discover -s tests -v
```

A new paid run uses the normal OpenRouter credential and optional guard-aware
base URL. The commands require explicit provider-cost ceilings:

```bash
python3 run_experiment.py \
  --live \
  --max-provider-cost-credits 0.30

python3 run_semantic_audit.py \
  --live \
  --max-provider-cost-credits 0.20
```

The frozen raw result is `results/experiment-raw-v1.json`; the full blinded
audit is `results/semantic-audit-v1.json`; and the independent manual findings
are `results/semantic-close-reading-v1.json`.

German and Chinese remain parameters, not predetermined winners. A web demo is
a later decision based on these results.

## License

MIT
