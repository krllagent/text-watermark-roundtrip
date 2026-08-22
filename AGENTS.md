# AGENTS.md

## Scope

This repository is the public, reproducible research artifact for
painintheagent Experiment 002: what happens to the reference SynthID Text
watermark after synonym edits, translation round trips, a full model
paraphrase and the DIPPER paraphraser, and what each method does to the
facts. Stage 1 (toy keyed lexical watermark, CPU-only) is kept as history;
the current experiment uses the `transformers` SynthID implementation, GPU
Pods on RunPod for corpus generation and DIPPER, OpenRouter for the
transformations and the blinded fact panel. It is not a production watermark
remover and not a web product; the demo lives in the painintheagent site.

## Evidence rules

- Never describe the detector as Claude, Gemini, or SynthID verification.
- Keep `densityBps` tied to eligible synonym positions and always report
  realized coverage over all words.
- Treat exact integer counts and fractions as authoritative. Floats are display
  values only.
- Preserve `insufficient_evidence`; never convert it to `z=0` or silently drop
  it from aggregates.
- Wrong keys are null controls, not bootstrap samples. Later confidence
  intervals resample paired document IDs.
- Transformations must not receive the marker key, density, or encoder
  lexicon.
- Maintained JSON data includes `schemaVersion`, `verifiedAt`, `methodology`,
  and `sources`.

## Engineering rules

- Local analysis depends only on `requirements.txt` (NumPy, PyTorch CPU,
  Transformers, Hugging Face Hub, PyYAML); GPU jobs pin their own stacks in
  the RunPod runners. Do not add dependencies without updating
  `requirements.txt` and `THIRD_PARTY_NOTICES.md`.
- RunPod runners never expose Pod files without the per-run bearer token
  (`control_server.py`) and never put credentials in process argv; the
  delete watchdog reads them through `EnvironmentFile=`.
- Tests first for marker, detector, protected-token, runner, or serialization
  changes.
- Analyze an immutable token stream before applying edits right-to-left.
- Keep HMAC framing and golden vectors backward compatible within a scheme
  version. A breaking change requires a new version and fixtures.
- `text_contract.py` owns protected-span behavior for both the marker and later
  transformations.
- No paid model calls before the corpus, config, key derivation, prompts, and
  control artifacts are frozen.

## Commands

```bash
python3 -m unittest discover -s tests -v   # full suite; includes slow recomputations
python3 -m compileall -q .
python3 -m ruff check .
```

Paid or GPU stages are never part of the test suite. Re-running them needs
`OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` and `RUNPOD_API_KEY` /
`RUNPOD_API_BASE_URL` in the environment (an Agent API Guard base URL with a
fake key works; a real key works the same way).
