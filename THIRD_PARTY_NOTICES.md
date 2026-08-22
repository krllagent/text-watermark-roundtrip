# Third-party notices

The code in this repository is MIT-licensed (see `LICENSE`). It depends on and
exercises third-party components under their own terms:

| Component | Use here | License (as published by the owner) |
|---|---|---|
| SynthID Text reference implementation (Google DeepMind), via `transformers` | watermark application and detection | Apache-2.0 (code); method described in Nature, 2024 |
| Qwen2.5-0.5B-Instruct, Qwen2.5-14B-Instruct (Alibaba) | corpus generation | Apache-2.0 (model cards) |
| Qwen3.7 Plus via OpenRouter (Alibaba) | synonym / paraphrase / translation transformations | provider terms of service |
| `kalpeshk2011/dipper-paraphraser-xxl` (DIPPER-11B) | paraphrase attack baseline | Apache-2.0 (model card) |
| GPT-5.6 Luna, Claude Sonnet 5, Gemma 4 31B IT, Grok 4.6 via OpenRouter | blinded fact panel | provider terms of service |
| P-SP embedding model from the DIPPER authors | semantic similarity metric | as published by the authors |
| PyTorch, Transformers, Hugging Face Hub, NumPy, PyYAML | runtime | BSD / Apache-2.0 |

Generated texts and result tables in `results/` are synthetic research
artifacts produced by the models above under the prompts in `configs/`; they
are released with this repository for reproduction and audit. The author does
not claim rights over third-party model outputs beyond what those licenses and
terms allow.
