"""Reproduce the published DIPPER baseline on the small SynthID smoke corpus.

The local ``prepare`` and ``analyze`` commands use only repository inputs.  The
``remote`` command is self-contained and is intended to run on one CUDA GPU
after installing the pinned third-party inference dependencies.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "results" / "synthid-corpus-gpu-v2.json"
DEFAULT_PRIOR_SMOKE = ROOT / "results" / "synthid-smoke-v1.json"
DEFAULT_INPUT = ROOT / "results" / "dipper-smoke-inputs-v1.json"
DEFAULT_REMOTE_OUTPUT = ROOT / "results" / "dipper-smoke-raw-v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "dipper-smoke-v1.json"

MODEL_ID = "kalpeshk2011/dipper-paraphraser-xxl"
MODEL_REVISION = "c1fbf7a958a2aab022e9e6f81f7a3139f9e6ee3c"
TOKENIZER_ID = "google/t5-v1_1-xxl"
TOKENIZER_REVISION = "3db67ab1af984cf10548a73467f0e5bca2aaaeb2"
TRANSFORMERS_VERSION = "4.40.2"
NLTK_VERSION = "3.8.1"
LEXICAL_DIVERSITY = 60
ORDER_DIVERSITY = 20
SENTENCE_CHUNK_SIZE = 3
PROMPT_MAX_TOKENS = 1_600
RESPONSE_MAX_TOKENS = 1_600
SEED = 123
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 1.0
PUBLISHED_THRESHOLD = 0.5067
DEFAULT_DOCUMENT_IDS = ("doc-01", "doc-04")
PRIOR_METHODS = ("synonyms", "paraphrase", "roundtrip-de", "roundtrip-zh")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(payload)


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evidence_sources() -> list[dict[str, str]]:
    return [
        {
            "title": "ETH SynthID-Text scrubbing evaluation",
            "url": "https://www.sri.inf.ethz.ch/blog/probingsynthid",
        },
        {
            "title": "ETH DIPPER scrubbing configuration",
            "url": (
                "https://github.com/eth-sri/watermark-stealing/blob/main/configs/"
                "scrubbing/llama7b/dipper_dolly_gptwm.yaml"
            ),
        },
        {
            "title": "ETH DIPPER model configuration",
            "url": (
                "https://github.com/eth-sri/watermark-stealing/blob/main/"
                "configs/model/dipper.yaml"
            ),
        },
        {
            "title": "ETH DIPPER attack implementation",
            "url": (
                "https://github.com/eth-sri/watermark-stealing/blob/main/"
                "src/attackers/our_attacker.py"
            ),
        },
        {
            "title": "Official DIPPER implementation and model instructions",
            "url": "https://github.com/martiansideofthemoon/ai-detection-paraphrases",
        },
        {
            "title": "DIPPER-11B model",
            "url": "https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl",
        },
    ]


def attack_contract() -> dict[str, object]:
    return {
        "doSample": True,
        "lexicalDiversity": LEXICAL_DIVERSITY,
        "logitsProcessors": [],
        "maxNewTokens": RESPONSE_MAX_TOKENS,
        "maxPromptTokens": PROMPT_MAX_TOKENS,
        "model": MODEL_ID,
        "modelRevision": MODEL_REVISION,
        "numBeams": 1,
        "orderDiversity": ORDER_DIVERSITY,
        "publishedRepetitionPenaltyConfig": 1.6,
        "reseedBeforeEverySentenceChunk": True,
        "seed": SEED,
        "sentenceChunkSize": SENTENCE_CHUNK_SIZE,
        "temperature": TEMPERATURE,
        "tokenizer": TOKENIZER_ID,
        "tokenizerRevision": TOKENIZER_REVISION,
        "topK": TOP_K,
        "topP": TOP_P,
    }


def dipper_prompt(
    *,
    prefix: str,
    sentence_window: str,
    lexical_diversity: int = LEXICAL_DIVERSITY,
    order_diversity: int = ORDER_DIVERSITY,
) -> str:
    """Build the control-code prompt used by the published DIPPER code."""
    lex_code = 100 - lexical_diversity
    order_code = 100 - order_diversity
    result = f"lexical = {lex_code}, order = {order_code}"
    normalized_prefix = " ".join(prefix.replace("\n", " ").split())
    if normalized_prefix:
        result += f" {normalized_prefix}"
    return result + f" <sent> {sentence_window} </sent>"


def build_input_artifact(
    *,
    corpus: Mapping[str, object],
    corpus_sha256: str,
    smoke: Mapping[str, object],
    smoke_sha256: str,
    document_ids: Sequence[str] = DEFAULT_DOCUMENT_IDS,
    verified_at: str | None = None,
) -> dict[str, object]:
    now = verified_at or utc_now()
    corpus_documents = corpus.get("documents")
    smoke_documents = smoke.get("documents")
    if not isinstance(corpus_documents, list) or not isinstance(smoke_documents, list):
        raise ValueError("corpus and smoke artifacts must contain document arrays")
    by_id = {
        str(document["documentId"]): document
        for document in corpus_documents
        if isinstance(document, Mapping)
    }
    cases: list[dict[str, object]] = []
    for document_id in document_ids:
        document = by_id.get(document_id)
        if not isinstance(document, Mapping):
            raise ValueError(f"unknown corpus document {document_id}")
        marked = document.get("marked")
        if not isinstance(marked, Mapping):
            raise ValueError(f"corpus document {document_id} has no marked lane")
        source_text = str(marked["text"])
        prefix = str(marked["prompt"])
        cases.append(
            {
                "caseId": f"{document_id}::marked-source",
                "documentId": document_id,
                "inputKind": "marked-source",
                "inputText": source_text,
                "inputTextSha256": sha256_text(source_text),
                "methodBeforeDipper": None,
                "preDipperMeanG": float(marked["meanG"]),
                "prefix": prefix,
                "prefixSha256": sha256_text(prefix),
            }
        )

    for document_id in document_ids:
        source = str(by_id[document_id]["marked"]["text"])  # type: ignore[index]
        prefix = str(by_id[document_id]["marked"]["prompt"])  # type: ignore[index]
        rows = [
            row
            for row in smoke_documents
            if isinstance(row, Mapping) and str(row.get("documentId")) == document_id
        ]
        by_method = {str(row.get("method")): row for row in rows}
        for method in PRIOR_METHODS:
            row = by_method.get(method)
            if not isinstance(row, Mapping):
                raise ValueError(f"missing prior smoke row {document_id}::{method}")
            if str(row.get("sourceText")) != source:
                raise ValueError(
                    f"prior smoke source does not match corpus for {document_id}::{method}"
                )
            detector = row.get("transformedDetector")
            if not isinstance(detector, Mapping) or detector.get("meanG") is None:
                raise ValueError(f"prior smoke detector is missing for {document_id}::{method}")
            text = str(row["evaluatedOutputText"])
            cases.append(
                {
                    "caseId": f"{document_id}::{method}",
                    "documentId": document_id,
                    "inputKind": "prior-smoke-output",
                    "inputText": text,
                    "inputTextSha256": sha256_text(text),
                    "methodBeforeDipper": method,
                    "preDipperMeanG": float(detector["meanG"]),
                    "prefix": prefix,
                    "prefixSha256": sha256_text(prefix),
                }
            )

    expected = len(document_ids) * (1 + len(PRIOR_METHODS))
    if len(cases) != expected or len({case["caseId"] for case in cases}) != expected:
        raise ValueError("DIPPER input cases are incomplete or duplicated")
    return {
        "attack": attack_contract(),
        "cases": cases,
        "createdAt": now,
        "inputs": {
            "corpusSha256": corpus_sha256,
            "documentIds": list(document_ids),
            "priorSmokeSha256": smoke_sha256,
        },
        "methodology": (
            "Two direct marked-source attacks provide the scientific comparison. "
            "The eight prior smoke outputs are a secondary, two-stage diagnostic. "
            "Every case uses the original generation request as DIPPER context and "
            "the published lexical-diversity 60, order-diversity 20, three-sentence "
            "chunking, temperature 0.7, one-beam sampling path with seed 123 reset "
            "before every chunk. No watermark key, detector output, or sampling table "
            "is provided to DIPPER."
        ),
        "schemaVersion": 1,
        "sources": evidence_sources(),
        "verifiedAt": now,
    }


def _remote_checkpoint(
    *,
    input_artifact: Mapping[str, object],
    input_sha256: str,
    completed_cases: Sequence[Mapping[str, object]],
    runtime: Mapping[str, object],
    status: str,
    started_at: str,
) -> dict[str, object]:
    return {
        "attack": input_artifact["attack"],
        "cases": list(completed_cases),
        "createdAt": started_at,
        "inputSha256": input_sha256,
        "methodology": (
            "Run the public DIPPER-11B checkpoint in float32 using the ETH baseline "
            "prompt construction. Normalize whitespace, sentence-tokenize with NLTK, "
            "rewrite consecutive three-sentence windows, and append each generated "
            "window to the prefix used for subsequent windows. Persist a checkpoint "
            "after every completed case."
        ),
        "runtime": dict(runtime),
        "schemaVersion": 1,
        "sources": evidence_sources(),
        "status": status,
        "verifiedAt": utc_now(),
    }


def run_remote(input_path: Path, output_path: Path) -> None:
    """Run DIPPER inference. This function intentionally imports GPU deps lazily."""
    import nltk
    import torch
    import transformers
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    raw_input = input_path.read_bytes()
    input_artifact = json.loads(raw_input)
    if input_artifact.get("schemaVersion") != 1:
        raise ValueError("unsupported DIPPER input schema")
    if input_artifact.get("attack") != attack_contract():
        raise ValueError("DIPPER attack contract does not match this runner")
    cases = input_artifact.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("DIPPER input contains no cases")

    nltk.download("punkt", quiet=True)
    started_at = utc_now()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
    )
    model_load_started = time.monotonic()
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    model_load_seconds = time.monotonic() - model_load_started
    first_parameter = next(model.parameters())
    runtime = {
        "cudaRuntime": torch.version.cuda,
        "gpuName": torch.cuda.get_device_name(0),
        "modelDtype": str(first_parameter.dtype),
        "modelLoadSeconds": model_load_seconds,
        "nltk": nltk.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if first_parameter.dtype != torch.float32:
        raise RuntimeError(f"DIPPER model must be float32, got {first_parameter.dtype}")

    completed: list[dict[str, object]] = []
    input_sha256 = sha256_bytes(raw_input)
    write_json_atomic(
        output_path,
        _remote_checkpoint(
            input_artifact=input_artifact,
            input_sha256=input_sha256,
            completed_cases=completed,
            runtime=runtime,
            status="running",
            started_at=started_at,
        ),
    )
    for case_index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError("DIPPER case must be an object")
        case_started = time.monotonic()
        input_text = " ".join(str(case["inputText"]).split())
        sentences = nltk.sent_tokenize(input_text)
        if not sentences:
            raise ValueError(f"DIPPER case {case['caseId']} has no sentences")
        prefix = " ".join(str(case["prefix"]).replace("\n", " ").split())
        output_parts: list[str] = []
        chunk_records: list[dict[str, object]] = []
        for chunk_index, sentence_index in enumerate(
            range(0, len(sentences), SENTENCE_CHUNK_SIZE)
        ):
            window = " ".join(
                sentences[sentence_index : sentence_index + SENTENCE_CHUNK_SIZE]
            )
            final_prompt = dipper_prompt(prefix=prefix, sentence_window=window)
            untruncated = tokenizer(final_prompt, add_special_tokens=True)["input_ids"]
            encoded = tokenizer(
                [final_prompt],
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=PROMPT_MAX_TOKENS,
            )
            encoded = {name: tensor.to("cuda") for name, tensor in encoded.items()}
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            generation_started = time.monotonic()
            with torch.inference_mode():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=RESPONSE_MAX_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                    num_beams=1,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    top_p=TOP_P,
                )
            generation_seconds = time.monotonic() - generation_started
            generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            if not generated.strip():
                raise RuntimeError(
                    f"DIPPER produced an empty chunk for {case['caseId']}::{chunk_index}"
                )
            prefix += " " + generated
            output_parts.append(generated)
            chunk_record = {
                "chunkIndex": chunk_index,
                "generatedTokensIncludingDecoderStart": int(outputs.shape[-1]),
                "inputTokens": int(encoded["input_ids"].shape[-1]),
                "inputTruncated": len(untruncated) > PROMPT_MAX_TOKENS,
                "generationSeconds": generation_seconds,
                "promptSha256": sha256_text(final_prompt),
                "sentenceCount": len(
                    sentences[sentence_index : sentence_index + SENTENCE_CHUNK_SIZE]
                ),
                "untruncatedInputTokens": len(untruncated),
            }
            chunk_records.append(chunk_record)
            print(
                json.dumps(
                    {
                        "caseId": case["caseId"],
                        "chunk": chunk_index + 1,
                        "event": "dipper_chunk",
                        "inputTokens": chunk_record["inputTokens"],
                        "outputTokens": chunk_record[
                            "generatedTokensIncludingDecoderStart"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        output_text = " ".join(part.strip() for part in output_parts).strip()
        record = {
            "caseId": case["caseId"],
            "caseIndex": case_index,
            "chunks": chunk_records,
            "documentId": case["documentId"],
            "inputKind": case["inputKind"],
            "inputTextSha256": case["inputTextSha256"],
            "inputWordCount": len(input_text.split()),
            "methodBeforeDipper": case["methodBeforeDipper"],
            "outputText": output_text,
            "outputTextSha256": sha256_text(output_text),
            "outputWordCount": len(output_text.split()),
            "seconds": time.monotonic() - case_started,
            "sentenceCount": len(sentences),
        }
        completed.append(record)
        write_json_atomic(
            output_path,
            _remote_checkpoint(
                input_artifact=input_artifact,
                input_sha256=input_sha256,
                completed_cases=completed,
                runtime=runtime,
                status="running",
                started_at=started_at,
            ),
        )
        print(
            json.dumps(
                {
                    "caseId": case["caseId"],
                    "completedCases": len(completed),
                    "event": "dipper_case",
                    "outputWords": record["outputWordCount"],
                    "seconds": round(float(record["seconds"]), 2),
                    "totalCases": len(cases),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    write_json_atomic(
        output_path,
        _remote_checkpoint(
            input_artifact=input_artifact,
            input_sha256=input_sha256,
            completed_cases=completed,
            runtime=runtime,
            status="complete",
            started_at=started_at,
        ),
    )


def _summary_group(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    before = [float(row["beforeDetector"]["meanG"]) for row in rows]  # type: ignore[index]
    after = [float(row["afterDetector"]["meanG"]) for row in rows]  # type: ignore[index]
    distances = [float(row["wordDistanceFromDipperInput"]) for row in rows]
    return {
        "caseCount": len(rows),
        "meanAfterG": statistics.mean(after) if after else None,
        "meanBeforeG": statistics.mean(before) if before else None,
        "meanDeltaG": (
            statistics.mean(a - b for a, b in zip(after, before)) if after else None
        ),
        "meanWordDistanceFromDipperInput": (
            statistics.mean(distances) if distances else None
        ),
        "removedCount": sum(row.get("removedAfterDipper") is True for row in rows),
    }


def summarize_analyzed_cases(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    direct = [row for row in rows if row.get("inputKind") == "marked-source"]
    second_stage = [
        row for row in rows if row.get("inputKind") == "prior-smoke-output"
    ]
    methods = sorted(
        {
            str(row["methodBeforeDipper"])
            for row in second_stage
            if row.get("methodBeforeDipper") is not None
        }
    )
    return {
        "allCases": _summary_group(rows),
        "byPriorMethod": {
            method: _summary_group(
                [row for row in second_stage if row.get("methodBeforeDipper") == method]
            )
            for method in methods
        },
        "directMarkedSources": _summary_group(direct),
        "priorSmokeOutputs": _summary_group(second_stage),
    }


def analyzed_methodology(
    *, direct_count: int, second_stage_count: int, threshold: float
) -> str:
    return (
        "Score every DIPPER output locally with the exact SynthID sampling table "
        "serialized in the supplied GPU corpus. Compare it with the original marked "
        "text using exact tokenizer 5-gram reuse and classify removal at the explicitly "
        f"supplied mean-g threshold {threshold:.12g}. The artifact contains "
        f"{direct_count} direct marked-source cases and {second_stage_count} "
        "two-stage diagnostic cases."
    )


def analyze(
    *,
    input_path: Path,
    remote_path: Path,
    corpus_path: Path,
    output_path: Path,
    threshold: float,
) -> dict[str, object]:
    from run_experiment import fidelity_metrics
    from run_synthid_smoke import SynthIDDetector

    raw_input = input_path.read_bytes()
    input_artifact = json.loads(raw_input)
    remote = json.loads(remote_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if remote.get("status") != "complete":
        raise ValueError("remote DIPPER artifact is not complete")
    if remote.get("inputSha256") != sha256_bytes(raw_input):
        raise ValueError("remote DIPPER artifact points to a different input")
    input_cases = {
        str(case["caseId"]): case for case in input_artifact["cases"]
    }
    remote_cases = remote.get("cases")
    if not isinstance(remote_cases, list) or len(remote_cases) != len(input_cases):
        raise ValueError("remote DIPPER artifact has an incomplete case set")
    corpus_documents = {
        str(document["documentId"]): document for document in corpus["documents"]
    }
    detector = SynthIDDetector(corpus)
    rows: list[dict[str, object]] = []
    for remote_case in remote_cases:
        case_id = str(remote_case["caseId"])
        input_case = input_cases.get(case_id)
        if not isinstance(input_case, Mapping):
            raise ValueError(f"unknown remote DIPPER case {case_id}")
        if remote_case.get("inputTextSha256") != input_case.get("inputTextSha256"):
            raise ValueError(f"DIPPER input hash mismatch for {case_id}")
        document_id = str(input_case["documentId"])
        marked = corpus_documents[document_id]["marked"]
        marked_source = str(marked["text"])
        before_text = str(input_case["inputText"])
        after_text = str(remote_case["outputText"])
        before_detector = detector.score(before_text, marked_source=marked_source)
        after_detector = detector.score(after_text, marked_source=marked_source)
        reported_before = float(input_case["preDipperMeanG"])
        if abs(float(before_detector["meanG"]) - reported_before) > 1e-12:
            raise ValueError(f"pre-DIPPER detector mismatch for {case_id}")
        fidelity_from_input = fidelity_metrics(before_text, after_text)
        fidelity_from_marked = fidelity_metrics(marked_source, after_text)
        after_mean = after_detector.get("meanG")
        row = {
            "afterDetector": after_detector,
            "beforeDetector": before_detector,
            "caseId": case_id,
            "chunks": remote_case["chunks"],
            "dipperOutputText": after_text,
            "dipperSeconds": remote_case["seconds"],
            "documentId": document_id,
            "fidelityFromDipperInput": fidelity_from_input,
            "fidelityFromMarkedSource": fidelity_from_marked,
            "inputKind": input_case["inputKind"],
            "methodBeforeDipper": input_case["methodBeforeDipper"],
            "publishedThreshold": threshold,
            "removedAfterDipper": (
                after_mean is not None and float(after_mean) < threshold
            ),
            "scoreDeltaFromDipperInput": (
                None
                if after_mean is None
                else float(after_mean) - float(before_detector["meanG"])
            ),
            "wordDistanceFromDipperInput": float(
                fidelity_from_input["wordLevenshtein"]["normalizedDistance"]
            ),
            "wordDistanceFromMarkedSource": float(
                fidelity_from_marked["wordLevenshtein"]["normalizedDistance"]
            ),
        }
        rows.append(row)
    now = utc_now()
    artifact = {
        "attack": input_artifact["attack"],
        "createdAt": now,
        "documents": rows,
        "inputs": {
            "corpusSha256": sha256_file(corpus_path),
            "dipperInputSha256": sha256_file(input_path),
            "remoteArtifactSha256": sha256_file(remote_path),
        },
        "methodology": analyzed_methodology(
            direct_count=sum(row["inputKind"] == "marked-source" for row in rows),
            second_stage_count=sum(
                row["inputKind"] == "prior-smoke-output" for row in rows
            ),
            threshold=threshold,
        ),
        "publishedDetectionThreshold": threshold,
        "remoteRuntime": remote["runtime"],
        "schemaVersion": 1,
        "sources": evidence_sources(),
        "summary": summarize_analyzed_cases(rows),
        "verifiedAt": now,
    }
    write_json_atomic(output_path, artifact)
    return artifact


def _prepare(args: argparse.Namespace) -> int:
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    smoke = json.loads(args.prior_smoke.read_text(encoding="utf-8"))
    artifact = build_input_artifact(
        corpus=corpus,
        corpus_sha256=sha256_file(args.corpus),
        smoke=smoke,
        smoke_sha256=sha256_file(args.prior_smoke),
        document_ids=args.documents,
    )
    write_json_atomic(args.output, artifact)
    print(
        json.dumps(
            {
                "cases": len(artifact["cases"]),
                "event": "prepared",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _analyze(args: argparse.Namespace) -> int:
    artifact = analyze(
        input_path=args.input,
        remote_path=args.remote,
        corpus_path=args.corpus,
        output_path=args.output,
        threshold=args.threshold,
    )
    print(
        json.dumps(
            {
                "event": "analyzed",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "summary": artifact["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    prepare.add_argument("--prior-smoke", type=Path, default=DEFAULT_PRIOR_SMOKE)
    prepare.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    prepare.add_argument("--documents", nargs="+", default=list(DEFAULT_DOCUMENT_IDS))
    prepare.set_defaults(handler=_prepare)

    remote = commands.add_parser("remote")
    remote.add_argument("--input", type=Path, required=True)
    remote.add_argument("--output", type=Path, required=True)
    remote.set_defaults(
        handler=lambda args: (run_remote(args.input, args.output) is not None) and 1
    )

    analyze_command = commands.add_parser("analyze")
    analyze_command.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    analyze_command.add_argument("--remote", type=Path, default=DEFAULT_REMOTE_OUTPUT)
    analyze_command.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    analyze_command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    analyze_command.add_argument("--threshold", type=float, default=PUBLISHED_THRESHOLD)
    analyze_command.set_defaults(handler=_analyze)

    args = parser.parse_args(argv)
    result = args.handler(args)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
