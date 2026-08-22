# results/ map

Canonical artifacts of the current experiment are listed first; everything
else is an intermediate, superseded or stage-1 artifact kept for audit. No
file is deleted because later artifacts reference earlier ones by path and
sha256.

## Canonical (used by the article and the site tables)

- `quality-synthid-corpus-curated-v1.json`: Final 10-document corpus (Qwen2.5-14B-Instruct, A40): marked texts, clean twins, curation log, keys, generation metadata.
- `quality-synthid-curated-calibration-v1.json`: Detector calibration: 100k-table conditional nulls per text, pooled threshold 0.5095383 at 1% FPR, legacy-threshold audit.
- `curated-methods-v1.json`: Four API transformations (synonyms, paraphrase, DE and ZH round trip) of every marked text, with costs.
- `curated-dipper-v1.json`: DIPPER-11B paraphrases of every marked text (A100 80GB, pinned revision).
- `curated-percent-pairs-v1.json`: Per-pair detector scores, signal-removal percentages and exact 5-gram reuse.
- `curated-psp-v1.json`: P-SP similarity per pair (official model of the DIPPER authors).
- `curated-panel-xai-input-v1.json`: Frozen single-candidate panel input (50 batches, one candidate each) with blind map.
- `curated-panel-canary-single-v3.json`: Single-candidate canary (identical / tampered / empty) passed 4/4 by the final judges.
- `curated-panel-single-v1.json`: Final panel: 200 single-candidate verdicts (GPT-5.6 Luna, Claude Sonnet 5, Gemma 4 31B, Grok 4.6).
- `curated-percent-table-v2.json`: FINAL TABLE: per-method percentages and per-pair rows used by the article.
- `curated-panel-adjudication-v1.json`: Manual adjudication of every non-unanimous claim vote (Claude Code, two overturns confirmed by the operator).
- `exp004-ngram-retention-v1.json`: Proof of the first corpus failure: detector score vs exact source 5-gram reuse (r = 0.988) on the Qwen2.5-0.5B corpus.
- `synthid-corpus-v1.json`: First (rejected) corpus from Qwen2.5-0.5B-Instruct on RTX 4090: incoherent texts kept as evidence.
- `exp004-methods-v1.json`: Rewrites of the first corpus by six models; their literal copying explains the old 177/181 result.
- `curated-dipper-runpod-v2-lifecycle-v3.json`: RunPod lifecycle and cost record of the DIPPER run.
- `quality-synthid-runpod-lifecycle-v2.json`: RunPod lifecycle and cost record of the corpus generation.

## Superseded by the single-candidate re-judge (2026-08-22)

- `curated-percent-table-v1.json`: Superseded by v2: built on the five-candidate panel that cross-contaminated verdicts.
- `curated-panel-combined-v1.json`: Superseded: three five-candidate judges + single-candidate Grok 4.20.
- `curated-panel-three-vendor-v1.json`: Superseded five-candidate panel (GPT-5.6 Luna, Claude Haiku 4.5, Gemma 4).
- `curated-panel-xai-v1.json`: Superseded single-candidate Grok 4.20 verdicts (scored an empty text 100/100).
- `curated-panel-canary-single-v1.json`: Canary that disqualified Claude Haiku 4.5 and Grok 4.20.
- `curated-panel-canary-single-v2.json`: Canary attempt where Grok 4.6 rejected reasoning_effort=none (HTTP 400).

## Intermediate and stage-1 artifacts

- **anchor-retention** (1 files): Stage-1 anchor analysis.
  `anchor-retention-reanalysis-v1.json`
- **calibration-blind** (3 files): Stage-1 calibration blind review.
  `calibration-blind-mapping-v1.json`, `calibration-blind-packet-v1.json`, `calibration-blind-review-v1.json`
- **codex-close-reading** (1 files): Stage-1 close reading.
  `codex-close-reading-v5.json`
- **corpus-controls** (1 files): Stage-1 detector controls.
  `corpus-controls-v1.json`
- **curated-dipper-runpod** (7 files): Lifecycle records of DIPPER attempts (earlier attempts failed before a Pod started).
  `curated-dipper-runpod-lifecycle-v1.json`, `curated-dipper-runpod-lifecycle-v2.json`, `curated-dipper-runpod-lifecycle-v3.json`, `curated-dipper-runpod-lifecycle-v4.json`, `curated-dipper-runpod-lifecycle-v5.json`, `curated-dipper-runpod-v2-lifecycle-v1.json`, `curated-dipper-runpod-v2-lifecycle-v2.json`
- **curated-methods-checkpoint** (1 files): Checkpoint of the transformation run.
  `curated-methods-checkpoint-v1.json`
- **curated-panel** (6 files): Panel inputs/outputs and checkpoints.
  `curated-panel-adjudication-worklist-v1.json`, `curated-panel-combined-input-v1.json`, `curated-panel-input-v1.json`, `curated-panel-prior-three-vendor-v1.json`, `curated-panel-v1.json`, `curated-panel-v2.json`
- **curated-panel-canary** (8 files): Five-candidate canaries of the superseded panel.
  `curated-panel-canary-input-v1.json`, `curated-panel-canary-single-input-v1.json`, `curated-panel-canary-single-input-v2.json`, `curated-panel-canary-single-input-v3.json`, `curated-panel-canary-v1.json`, `curated-panel-canary-v2.json`, `curated-panel-canary-v3.json`, `curated-panel-canary-v4.json`
- **curated-psp-methods** (1 files): P-SP intermediate.
  `curated-psp-methods-v1.json`
- **dipper-runpod** (1 files): Lifecycle of the DIPPER smoke run.
  `dipper-runpod-lifecycle-v1.json`
- **dipper-smoke** (3 files): DIPPER smoke test on the first corpus.
  `dipper-smoke-inputs-v1.json`, `dipper-smoke-raw-v1.json`, `dipper-smoke-v1.json`
- **exp003** (6 files): Exp003: toy marker reruns with anchored judging.
  `exp003-anchored-judge-v1.json`, `exp003-anchored-v1.json`, `exp003-judge-rest-v1.json`, `exp003-judge-v1.json`, `exp003-marked-corpus-v1.json`, `exp003-methods-v1.json`
- **exp004** (2 files): Exp004: first real-SynthID attempt on the rejected 0.5B corpus.
  `exp004-combined-v1.json`, `exp004-judge-v1.json`
- **experiment** (2 files): Stage-1 raw experiment artifacts.
  `experiment-checkpoint-v1.json`, `experiment-raw-v1.json`
- **fidelity-recalculation** (1 files): Stage-1 fidelity recalculation.
  `fidelity-recalculation-v2.json`
- **final-holdout** (7 files): Toy-watermark holdout evaluations (stage 1).
  `final-holdout-controls-v8.json`, `final-holdout-controls-v9.json`, `final-holdout-v9-locked-v2-automated-v1.json`, `final-holdout-v9-locked-v2-blind-v1.json`, `final-holdout-v9-locked-v2-checkpoint-v1.json`, `final-holdout-v9-locked-v2-final-v1.json`, `final-holdout-v9-locked-v2-review-v1.json`
- **finalists** (3 files): Stage-1 finalist reviews.
  `finalists-blind-mapping-v1.json`, `finalists-blind-packet-v1.json`, `finalists-blind-review-v1.json`
- **judge-panel** (3 files): Stage-1 judge panels.
  `judge-panel-anchored-v1.json`, `judge-panel-models-v1.json`, `judge-panel-rematch-v1.json`
- **method-rematch** (1 files): Stage-1 method rematch.
  `method-rematch-v1.json`
- **model-canary** (18 files): Judge/rewriter model canaries from the earlier toy-watermark stage (Exp002 stage 1).
  `model-canary-checkpoint-v6.json`, `model-canary-luna-blind-v1.json`, `model-canary-luna-checkpoint-v1.json`, `model-canary-luna-final-v1.json`, `model-canary-luna-review-v1.json`, `model-canary-route-failure-v6.json`, `model-canary-terra-blind-v1.json`, `model-canary-terra-checkpoint-v1.json`, `model-canary-terra-final-v1.json`, `model-canary-terra-locked-blind-v1.json`, `model-canary-terra-locked-checkpoint-v1.json`, `model-canary-terra-locked-final-v1.json`, `model-canary-terra-locked-review-v1.json`, `model-canary-terra-locked-v2-blind-v1.json`, `model-canary-terra-locked-v2-checkpoint-v1.json`, `model-canary-terra-locked-v2-final-v1.json`, `model-canary-terra-locked-v2-review-v1.json`, `model-canary-terra-review-v1.json`
- **model-screen** (4 files): Model screening for the earlier stage.
  `model-screen-finalists-v1.json`, `model-screen-nonsignatories-v1.json`, `model-screen-qwen35-v1.json`, `model-screen-v2.json`
- **nonsignatory** (3 files): Stage-1 blinded reviews.
  `nonsignatory-blind-mapping-v1.json`, `nonsignatory-blind-packet-v1.json`, `nonsignatory-blind-review-v1.json`
- **paraphrase** (2 files): Stage-1 paraphrase runs.
  `paraphrase-across-models-v1.json`, `paraphrase-anchored-v1.json`
- **quality-conditioned-smoke** (1 files): Smoke test of quality-conditioned generation.
  `quality-conditioned-smoke-v1.json`
- **quality-synthid** (3 files): Corpus generation, calibration and lifecycle records.
  `quality-synthid-calibration-v1.json`, `quality-synthid-corpus-gpu-v1.json`, `quality-synthid-runpod-lifecycle-v1.json`
- **semantic-audit** (5 files): Stage-1 semantic audits.
  `semantic-audit-checkpoint-v1.json`, `semantic-audit-checkpoint-v4.json`, `semantic-audit-route-canary-failure-v1.json`, `semantic-audit-v1.json`, `semantic-audit-v4.json`
- **semantic-close-reading** (1 files): Stage-1 close reading.
  `semantic-close-reading-v1.json`
- **synthetic-preflight** (1 files): Stage-1 preflight.
  `synthetic-preflight-v1.json`
- **synthid** (5 files): First-corpus runs and smoke tests.
  `synthid-corpus-gpu-v1.json`, `synthid-corpus-gpu-v2.json`, `synthid-smoke-checkpoint-v1.json`, `synthid-smoke-v1.json`, `synthid-smoke-zdr-route-rejected-v1.json`
- **verified-paraphrase** (6 files): Earlier verified-paraphrase pipeline runs (stage 1, v4/v5).
  `verified-paraphrase-canary-v2.json`, `verified-paraphrase-checkpoint-v2.json`, `verified-paraphrase-checkpoint-v3.json`, `verified-paraphrase-checkpoint-v4.json`, `verified-paraphrase-derived-v5-single-pass.json`, `verified-paraphrase-raw-v4.json`
- **other** (2 files): Earlier iteration; see git log for context.
  `curated-dipper-inputs-v1.json`, `curated-dipper-raw-v1.json`
