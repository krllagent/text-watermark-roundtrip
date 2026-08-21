"""Generate a quality-gated SynthID corpus with a larger model on CUDA.

The remote job is intentionally self-contained so the exact source can be
embedded in an ephemeral RunPod Pod and hashed by the controller.  Every
accepted brief has one marked and one unmarked generation made from the same
prompt and seed.  A pair is retried as a unit when either side fails the
deterministic English-report quality contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import time
import unicodedata


DEFAULT_OUTPUT = Path("/workspace/quality-synthid-corpus-v1.json")
DEFAULT_CHECKPOINT = Path("/workspace/quality-synthid-corpus-v1.partial.json")

WATERMARK_KEYS = [
    654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468,
    990, 966, 226, 324, 585, 118, 504, 421, 521, 129, 669, 732, 225, 90, 960,
]
NGRAM_LEN = 5
CONTEXT_HISTORY_SIZE = 1024
SAMPLING_TABLE_SIZE = 65_536
SAMPLING_TABLE_SEED = 0

BRIEFS = [
    ("doc-01", "a public library replacing its paper visitor book with a tablet"),
    ("doc-02", "a bicycle courier service trialling a two-hour pickup window"),
    ("doc-03", "a school science fair judged by a rotating panel of parents"),
    ("doc-04", "a small bakery moving from telephone orders to a shared spreadsheet"),
    ("doc-05", "a community garden allocating plots for the coming season"),
    ("doc-06", "a museum trialling audio guides on visitors' own phones"),
    ("doc-07", "a repair cafe tracking appliances that return for a second repair"),
    ("doc-08", "a village bus route changing its evening timetable"),
    ("doc-09", "an office kitchen introducing a rota for its coffee machine"),
    ("doc-10", "a swimming pool changing how local clubs reserve lane time"),
]

PROMPT = (
    "Write a coherent English-language fictional report of 520 to 620 words about "
    "{brief}. Present it as an ordinary factual case report, not a story. Use seven "
    "to nine flowing prose paragraphs. Include at least two concrete calendar dates "
    "with a four-digit year, several internally consistent numbers, two named people, "
    "one cost written with a $ sign, one explicit deadline, a problem encountered, "
    "the measured outcome, and a final paragraph containing the exact words 'The "
    "report recommends'. Keep every fact internally consistent. Use plain natural "
    "English only. Do not use headings, lists, bullets, tables, fragments, invented "
    "technical jargon, or words from other writing systems. Finish the report "
    "naturally after the recommendation."
)

MONTH_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\b",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"\b[^\W_][\w'’-]*\b", re.UNICODE)
HEADING_OR_LIST_PATTERN = re.compile(
    r"(?m)^\s*(?:#{1,6}\s|[-*+]\s+|\d+[.)]\s+)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _letter_script_counts(text: str) -> tuple[int, int]:
    latin = 0
    non_latin = 0
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if "LATIN" in name:
            latin += 1
        else:
            non_latin += 1
    return latin, non_latin


def quality_issues(
    text: str,
    *,
    token_count: int,
    max_new_tokens: int,
    stopped_on_eos: bool,
) -> list[str]:
    """Return deterministic violations of the source-text contract."""
    issues: list[str] = []
    words = WORD_PATTERN.findall(text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    latin, non_latin = _letter_script_counts(text)
    letter_total = latin + non_latin

    if not 450 <= len(words) <= 700:
        issues.append("word_count")
    if not 6 <= len(paragraphs) <= 10:
        issues.append("paragraph_count")
    if not stopped_on_eos or token_count >= max_new_tokens:
        issues.append("no_natural_stop")
    if letter_total and non_latin / letter_total > 0.002:
        issues.append("non_latin_script")
    if HEADING_OR_LIST_PATTERN.search(text):
        issues.append("heading_or_list")
    if "\ufffd" in text or any(
        unicodedata.category(character) == "Cc" and character not in "\n\t\r"
        for character in text
    ):
        issues.append("invalid_character")
    if not re.search(r"\$\s?\d", text):
        issues.append("missing_currency")
    if not MONTH_PATTERN.search(text) or not re.search(r"\b20\d{2}\b", text):
        issues.append("missing_calendar_date")
    if "the report recommends" not in text.lower():
        issues.append("missing_recommendation")
    if not re.search(r"[.!?][\"'’)]?\s*$", text):
        issues.append("unfinished_sentence")
    return issues


def retry_prompt(base_prompt: str, previous_issues: list[str]) -> str:
    """Make a retry address prior mechanical failures without changing the brief."""
    if not previous_issues:
        return base_prompt
    reminders = {
        "word_count": "Keep the report between 520 and 620 words.",
        "paragraph_count": "Use exactly seven or eight prose paragraphs.",
        "no_natural_stop": "Finish naturally and do not continue past the recommendation.",
        "non_latin_script": "Use Latin-script English words only.",
        "heading_or_list": "Do not add any heading, label, number, or bullet before a paragraph.",
        "invalid_character": "Use ordinary printable prose characters only.",
        "missing_currency": "Include a concrete cost such as $4,250, with the $ sign present.",
        "missing_calendar_date": "Include a month name and a four-digit year in a date.",
        "missing_recommendation": (
            "Begin the final paragraph with the exact words 'The report recommends'."
        ),
        "unfinished_sentence": "End the final sentence with normal punctuation.",
    }
    selected = [reminders[issue] for issue in previous_issues if issue in reminders]
    return base_prompt + "\n\nMandatory corrections for this retry: " + " ".join(selected)


def _eos_ids(tokenizer, model) -> set[int]:
    values: set[int] = set()
    for candidate in (
        tokenizer.eos_token_id,
        getattr(model.generation_config, "eos_token_id", None),
    ):
        if isinstance(candidate, int):
            values.add(candidate)
        elif isinstance(candidate, (list, tuple)):
            values.update(int(value) for value in candidate)
    return values


def _watermark_processor(torch, processor_class, device):
    return processor_class(
        ngram_len=NGRAM_LEN,
        keys=WATERMARK_KEYS,
        sampling_table_size=SAMPLING_TABLE_SIZE,
        sampling_table_seed=SAMPLING_TABLE_SEED,
        context_history_size=CONTEXT_HISTORY_SIZE,
        device=device,
    )


def detector_trace(processor, tokenizer, text: str) -> dict[str, object]:
    import torch

    ids = tokenizer([text], return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.shape[1] <= NGRAM_LEN:
        return {
            "contextRepetitionMask": [],
            "detectorTokenIds": ids[0].tolist(),
            "gValues": [],
            "meanG": None,
            "perDepthMeanG": [],
            "validGValueCount": 0,
        }
    g_values = processor.compute_g_values(input_ids=ids)
    mask = processor.compute_context_repetition_mask(input_ids=ids)
    valid = mask.unsqueeze(-1).expand_as(g_values)
    valid_count = int(valid.sum().item())
    mean_g = (
        float((g_values * valid).sum().item() / valid_count)
        if valid_count
        else None
    )
    per_depth = []
    for depth in range(g_values.shape[-1]):
        count = int(mask.sum().item())
        per_depth.append(
            float((g_values[:, :, depth] * mask).sum().item() / count)
            if count
            else None
        )
    return {
        "contextRepetitionMask": mask[0].tolist(),
        "detectorTokenIds": ids[0].tolist(),
        "gValues": g_values[0].to(torch.uint8).tolist(),
        "meanG": mean_g,
        "perDepthMeanG": per_depth,
        "validGValueCount": valid_count,
    }


def generate_one(
    *,
    torch,
    tokenizer,
    model,
    processor_class,
    canonical_sampling_table,
    prompt: str,
    marked: bool,
    seed: int,
    settings: dict[str, object],
) -> dict[str, object]:
    from transformers import LogitsProcessorList

    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer([formatted_prompt], return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": True,
        "max_new_tokens": int(settings["maxNewTokens"]),
        "temperature": float(settings["temperature"]),
        "top_k": int(settings["topK"]),
        "top_p": float(settings["topP"]),
        "repetition_penalty": float(settings["repetitionPenalty"]),
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if marked:
        generation_processor = _watermark_processor(
            torch, processor_class, model.device
        )
        generation_processor.sampling_table.copy_(
            canonical_sampling_table.to(model.device)
        )
        kwargs["logits_processor"] = LogitsProcessorList([generation_processor])
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**kwargs)
    elapsed = time.monotonic() - started
    generated_ids = output[0, input_ids.shape[1] :].detach().cpu().tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    eos_ids = _eos_ids(tokenizer, model)
    stopped_on_eos = bool(generated_ids and generated_ids[-1] in eos_ids)
    issues = quality_issues(
        text,
        token_count=len(generated_ids),
        max_new_tokens=int(settings["maxNewTokens"]),
        stopped_on_eos=stopped_on_eos,
    )
    return {
        "formattedPrompt": formatted_prompt,
        "generationTokenIds": generated_ids,
        "inputTokenIds": encoded["input_ids"][0].tolist(),
        "marked": marked,
        "qualityIssues": issues,
        "seconds": elapsed,
        "seed": seed,
        "stoppedOnEos": stopped_on_eos,
        "stopTokenId": generated_ids[-1] if generated_ids else None,
        "text": text,
        "textSha256": sha256_bytes(text.encode("utf-8")),
        "tokenCount": len(generated_ids),
        "wordCount": len(WORD_PATTERN.findall(text)),
    }


def _artifact_shell(config: dict[str, object], sampling_table: list[int]) -> dict[str, object]:
    return {
        "contextHistorySize": CONTEXT_HISTORY_SIZE,
        "createdAt": utc_now(),
        "documents": [],
        "generation": config["generation"],
        "keys": WATERMARK_KEYS,
        "methodology": (
            "Quality-gated paired marked and unmarked English reports generated from "
            "identical prompts and seeds on one CUDA GPU. Each pair is retried as a "
            "unit when either output violates deterministic language, length, natural-"
            "stop, structure, date, currency, or recommendation checks. The canonical "
            "CPU SynthID sampling table is copied byte-for-byte into every fresh CUDA "
            "generation processor, and full model-free detector traces are retained."
        ),
        "model": config["model"],
        "modelRevision": config["modelRevision"],
        "ngramLen": NGRAM_LEN,
        "qualityContract": {
            "allowedParagraphs": [6, 10],
            "allowedWords": [450, 700],
            "maxNonLatinLetterFraction": 0.002,
            "naturalEosRequired": True,
            "required": [
                "calendar date",
                "currency with $ sign",
                "exact recommendation phrase",
            ],
        },
        "samplingTable": sampling_table,
        "schemaVersion": 1,
        "sources": [
            "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct",
            "https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.SynthIDTextWatermarkLogitsProcessor",
            "https://doi.org/10.1038/s41586-024-08025-4",
        ],
        "status": "running",
    }


def run(config: dict[str, object], output_path: Path, checkpoint_path: Path) -> dict[str, object]:
    import huggingface_hub
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        SynthIDTextWatermarkLogitsProcessor,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model_id = str(config["model"])
    revision = str(config["modelRevision"])
    generation = dict(config["generation"])
    max_attempts = int(config["maxAttemptsPerPair"])
    document_count = int(config["documentCount"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    detector = _watermark_processor(
        torch, SynthIDTextWatermarkLogitsProcessor, torch.device("cpu")
    )
    canonical_sampling_table = detector.sampling_table.detach().cpu().clone()
    artifact = _artifact_shell(config, canonical_sampling_table.tolist())
    artifact["runtime"] = {
        "cudaRuntime": torch.version.cuda,
        "cudnnVersion": torch.backends.cudnn.version(),
        "gpuName": torch.cuda.get_device_name(0),
        "huggingfaceHubVersion": huggingface_hub.__version__,
        "modelDtype": str(model.dtype),
        "platform": platform.platform(),
        "pythonVersion": platform.python_version(),
        "torchVersion": torch.__version__,
        "transformersVersion": transformers.__version__,
    }
    write_json_atomic(checkpoint_path, artifact)

    for index, (document_id, brief) in enumerate(BRIEFS[:document_count]):
        prompt = PROMPT.format(brief=brief)
        attempts: list[dict[str, object]] = []
        accepted: tuple[dict[str, object], dict[str, object]] | None = None
        for attempt in range(max_attempts):
            seed = int(config["seedBase"]) + index * max_attempts + attempt
            prior_issues: list[str] = []
            if attempts:
                prior_issues = sorted(
                    set(attempts[-1]["markedIssues"])
                    | set(attempts[-1]["unmarkedIssues"])
                )
            attempt_prompt = retry_prompt(prompt, prior_issues)
            marked = generate_one(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                processor_class=SynthIDTextWatermarkLogitsProcessor,
                canonical_sampling_table=canonical_sampling_table,
                prompt=attempt_prompt,
                marked=True,
                seed=seed,
                settings=generation,
            )
            unmarked = generate_one(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                processor_class=SynthIDTextWatermarkLogitsProcessor,
                canonical_sampling_table=canonical_sampling_table,
                prompt=attempt_prompt,
                marked=False,
                seed=seed,
                settings=generation,
            )
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "markedIssues": marked["qualityIssues"],
                    "markedTextSha256": marked["textSha256"],
                    "seed": seed,
                    "unmarkedIssues": unmarked["qualityIssues"],
                    "unmarkedTextSha256": unmarked["textSha256"],
                }
            )
            print(
                json.dumps(
                    {
                        "attempt": attempt + 1,
                        "documentId": document_id,
                        "event": "pair_attempt",
                        "markedIssues": marked["qualityIssues"],
                        "unmarkedIssues": unmarked["qualityIssues"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not marked["qualityIssues"] and not unmarked["qualityIssues"]:
                accepted = marked, unmarked
                break
        if accepted is None:
            artifact["status"] = "quality_gate_failed"
            artifact["failedDocumentId"] = document_id
            artifact["failedAttempts"] = attempts
            artifact["verifiedAt"] = utc_now()
            write_json_atomic(output_path, artifact)
            raise RuntimeError(f"quality gate failed for {document_id}")

        marked, unmarked = accepted
        for row in (marked, unmarked):
            trace = detector_trace(detector, tokenizer, str(row["text"]))
            row.update(trace)
            comparable = list(row["generationTokenIds"])
            while comparable and comparable[-1] in tokenizer.all_special_ids:
                comparable.pop()
            row["decodeRetokenizationExact"] = comparable == trace["detectorTokenIds"]
            row["prompt"] = attempt_prompt
        document = {
            "attempts": attempts,
            "brief": brief,
            "documentId": document_id,
            "marked": marked,
            "unmarked": unmarked,
        }
        artifact["documents"].append(document)
        write_json_atomic(checkpoint_path, artifact)
        print(
            json.dumps(
                {
                    "attempts": len(attempts),
                    "documentId": document_id,
                    "event": "pair_accepted",
                    "markedMeanG": marked["meanG"],
                    "unmarkedMeanG": unmarked["meanG"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    documents = artifact["documents"]
    artifact["markedMeanG"] = statistics.mean(
        float(document["marked"]["meanG"]) for document in documents
    )
    artifact["unmarkedMeanG"] = statistics.mean(
        float(document["unmarked"]["meanG"]) for document in documents
    )
    artifact["status"] = "complete"
    artifact["verifiedAt"] = utc_now()
    write_json_atomic(output_path, artifact)
    write_json_atomic(checkpoint_path, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run(config, args.output, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
