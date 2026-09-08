# Text Watermark Round-Trip

Does a full paraphrase remove Google DeepMind's **SynthID Text** watermark, and
what does it do to the facts? A small, fully logged experiment behind the
[Pain in the Agent article on text-watermark removal](https://painintheagent.com/blog/text-watermark-removal-retest/)
and the [AI text watermark remover](https://painintheagent.com/tools/ai-text-watermark-remover/).

**Follow-up review:** the main result remains useful, with a substantive
correction to the generator order. The original implementation placed SynthID
before temperature/top-k/top-p. Standard Transformers 5.15.1 places it after
those transforms and before the token is sampled.

| Workflow | Original A | Repeat B | New corpus C, original order | Standard-order control |
|---|---:|---:|---:|---:|
| Protected single-pass LLM paraphrase | 10/10 | 8/10 | 10/10 | 10/10 |
| DIPPER as configured | 10/10 | 10/10 | 10/10 | 9/10 |
| Light synonyms | 2/10 | 0/10 | 1/10 | 1/10 |
| Translation via German | 0/10 | 0/10 | 0/10 | 0/10 |
| Translation via Chinese | 0/10 | 0/10 | 1/10 | 5/10 |

These are outputs below a prespecified local-key threshold, out of ten attempts
per method. A and B each contain one failed Chinese translation. The accepted
original threshold stays 0.5095383054287164; C and its control share
0.509656862745098, fixed before transformations. Their sources differ despite
matching briefs, key and seed schedule. They are not forty independent documents.

The original four-model panel credited the LLM workflow with 100/100 selected
claims. The later unblinded review, including three documented author
adjudications, retains 97/100 for paraphrase and 74/100 for DIPPER in the
standard-order control. Input protection differs: API methods mask and restore
selected spans, while DIPPER receives whitespace-normalized raw text. This is
a comparison of those workflows, not an isolated model-quality benchmark.

The [updated article](https://painintheagent.com/blog/text-watermark-removal-retest/#revalidation-title)
and [additional evidence](https://painintheagent.com/research/synthid-revalidation/v1/summary.json)
carry the complete scope and source/candidate records. The historical
`results/curated-percent-table-v2.json` and its exports remain unchanged.
The first failed experiment is also retained in the repository.

This measures a local reference SynthID configuration. It does not verify
private Claude/Gemini keys or the current multistage web tool, which differs
from the measured Qwen3.7 Plus single-pass workflow.

## Try it

[AI text watermark remover](https://painintheagent.com/tools/ai-text-watermark-remover/)

## Cite

[CITATION.cff](CITATION.cff)

## What is in the box

| Stage | Script | Where it ran | Cost |
|---|---|---|---|
| Corpus: 10 marked reports + 10 clean twins, `Qwen/Qwen2.5-14B-Instruct` fp16, SynthID via `transformers` | `gen_quality_synthid_corpus_gpu.py` driven by `run_quality_corpus_on_runpod.py`, curation `curate_synthid_corpus.py` | RunPod, NVIDIA A40, $0.44/h, two runs ≈ 32 min | < $0.25 |
| Detector calibration: 100k random-table conditional nulls per text, pooled threshold 0.5095383 at a 1% conditional-null target | `calibrate_synthid_threshold.py` | CPU | $0 |
| Four transformations (synonyms, full paraphrase, DE and ZH round trip) with `qwen/qwen3.7-plus`, temperature 0 | `run_synthid_smoke.py` | OpenRouter | $0.07 |
| DIPPER-11B paraphrase (`kalpeshk2011/dipper-paraphraser-xxl`, pinned revision, fp32) | `dipper_smoke.py` driven by `run_dipper_on_runpod_v2.py` | RunPod, A100 80GB, $1.39/h, ≈ 6 min | $0.13 |
| Pairs, signal removal, exact 5-gram reuse, P-SP | `curated_percent_eval.py build-pairs`, `compute_curated_psp.py` | CPU | $0 |
| Blinded fact panel: 10 frozen claims per source, four vendors, one candidate per prompt, canary with identical / tampered / empty text | `run_curated_panel.py` | OpenRouter | $0.92 (+ $0.08 canaries) |
| Final table and manual adjudication of non-unanimous votes | `curated_percent_eval.py finalize`, `adjudication_worklist.py` | CPU | $0 |

Total paid for the final table: about $1.45. The exact lifecycle and cost
records are in `results/*lifecycle*.json` and in every paid artifact's
`budget` / `totalCostUsd` fields.

`results/README.md` is the map: which 16 files are canonical, which 6 are
superseded by the 2026-08-22 re-judge, and what the other ~110 intermediate and
stage-1 artifacts are. Nothing is deleted because artifacts reference each
other by path and sha256.

## Reproduce the table without paying anything

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # CPU torch is enough; see the file
python3 -m unittest tests.test_curated_percent_eval tests.test_run_curated_panel \
        tests.test_control_server tests.test_derive_curated_eval_v5
python3 curated_percent_eval.py finalize \
  --config configs/curated-percent-eval-v5.json \
  --pairs results/curated-percent-pairs-v1.json \
  --psp results/curated-psp-v1.json \
  --panel-input results/curated-panel-xai-input-v1.json \
  --panel-output results/curated-panel-single-v1.json \
  --methods results/curated-methods-v1.json \
  --output /tmp/curated-percent-table-v2.json
diff <(python3 -c "import json;print(json.dumps(json.load(open('/tmp/curated-percent-table-v2.json'))['summary'],sort_keys=True,indent=1))") \
     <(python3 -c "import json;print(json.dumps(json.load(open('results/curated-percent-table-v2.json'))['summary'],sort_keys=True,indent=1))")
```

The complete legacy suite also expects historical prerequisites absent from
the frozen source artifact, including `model-canary-terra-locked-final-v1.json`.
Use the listed current-result checks to reproduce this table; the missing
legacy files are not proof that the current numerical results failed.

## Re-run the paid and GPU stages

Set `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` and `RUNPOD_API_KEY` /
`RUNPOD_API_BASE_URL`. Every runner freezes its prompts, config hashes and cost
caps before the first call, checkpoints each paid response, and refuses to
exceed the budget written in the config. GPU jobs run on RunPod Pods: the
runner creates the Pod, ships the job script through the Pod environment,
polls a token-protected control channel (`control_server.py`; the RunPod HTTP
proxy is public, so the per-run bearer token is required for every download),
downloads the artifact, deletes the Pod, and arms a `systemd-run --user`
watchdog that deletes the Pod even if the controller dies (credentials reach
the watchdog through a 0600 `EnvironmentFile`, never argv).

| Job | Command | Needs |
|---|---|---|
| Corpus | `python3 run_quality_corpus_on_runpod.py run` | 1× A40-class GPU (≥ 40 GB), ~30 min |
| Transformations | `python3 run_synthid_smoke.py --corpus results/quality-synthid-corpus-curated-v1.json --model qwen/qwen3.7-plus --output results/curated-methods-v1.json` | OpenRouter |
| DIPPER | `python3 run_dipper_on_runpod_v2.py --gpu-types "NVIDIA A100 80GB PCIe"` | 1× A100 80GB (fp32 11B), ~6 min |
| Panel canary, then panel | `python3 run_curated_panel.py prepare-canary --config configs/curated-percent-eval-v5.json --single --output results/curated-panel-canary-single-input-v3.json` → `run --mode canary` → `run --mode full --input results/curated-panel-xai-input-v1.json` | OpenRouter |

## Method notes and limitations

- **Detector.** Mean g-value over the exact SynthID sampling table with our
  key (ngram 5, table 65,536, depth 30); no language model runs at detection
  time. One pooled threshold for all texts, calibrated before any method ran;
  the 1% target is conditional on the simulated null, not a measured population
  FPR. A per-document length-aware threshold is also stored in the calibration
  artifact. One paraphrase lies within 0.0002 of the pooled threshold.
- **Signal removed %** is the share of the distance between the marked text's
  score and its clean twin's score that the candidate travelled. It is not a
  share of "AI-written text".
- **Judges.** A first panel showed three judges five candidates of one
  document per prompt; verdicts leaked between candidates (one judge charged
  DIPPER's changed numbers to the plain paraphrase) and Grok 4.20 scored an
  empty text 100/100. It was replaced by single-candidate prompts and a canary
  that disqualified Claude Haiku 4.5 and Grok 4.20; the final judges are
  GPT-5.6 Luna, Claude Sonnet 5, Gemma 4 31B IT and Grok 4.6 (reasoning low).
  Both panels are kept; only the second feeds the published numbers.
- **Scope.** The original marked corpus contains ten fictional English reports of
  480–639 words, with a 450–700 word generation quality gate, one key and one
  corpus. The follow-up batches above broaden that test but remain small. Short texts carry less signal; production keys are private; the
  repo says nothing about whether a specific Gemini output was cleaned.
- **Rewriter choice.** Removal only works if the rewriting model does not add
  an equivalent watermark itself. The API model used here did not add this
  mark; open weights under your own control are the safer default.

## Layout

`configs/` frozen designs (prompts, judges, budgets, claims) · `corpus/` stage-1
texts · `results/` every artifact (see its README) · `tests/` unit tests ·
`docs/plans/` working plans · `scripts/` operator helpers (the Guard setup
script is specific to the author's host and not needed for reproduction) ·
`attic/` stage-1 leftovers.

## Citing and licenses

Code: MIT (`LICENSE`). Third-party models, code and terms: `THIRD_PARTY_NOTICES.md`.
Citation metadata: `CITATION.cff`. The watermark scheme is Dathathri et al.,
*Nature* 2024; the paraphrase attack baseline is Krishna et al. 2023 (DIPPER);
the ETH Zurich SRI probing post (Jovanović, Gloaguen, Vechev, 2024) is the
closest prior measurement.
