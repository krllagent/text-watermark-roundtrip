# AGENTS.md

## Scope

This repository is the public, reproducible research artifact for
painintheagent Experiment 002. Stage 1 is a CPU-only toy watermark experiment,
not a production watermark remover and not a web product.

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

- Python standard library only unless Kirill explicitly changes scope.
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
python3 -m unittest discover -s tests -v
python3 -m compileall -q .
```
