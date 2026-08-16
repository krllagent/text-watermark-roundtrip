"""Run deterministic CPU controls for the transparent toy detector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from text_contract import TEXT_CONTRACT_VERSION, TOKENIZER_VERSION
from watermark_toy import (
    DEFAULT_CONTEXT_WIDTH,
    DEFAULT_MIN_ACTIVE_POSITIONS,
    DETECTION_ALPHA,
    SCHEME_VERSION,
    Document,
    SynonymLexicon,
    encode_text,
    load_lexicon,
    run_wrong_key_controls,
    score_corpus,
)


DEFAULT_WRONG_KEY_SEED = b"exp002-synthetic-preflight-v1"


def run_preflight(
    *,
    fixture: dict[str, object],
    fixture_sha256: str,
    lexicon: SynonymLexicon,
    key: bytes,
    density_bps: int,
    wrong_key_count: int,
    wrong_key_seed: bytes = DEFAULT_WRONG_KEY_SEED,
) -> dict[str, object]:
    documents = build_synthetic_documents(fixture)
    generator = fixture["generator"]
    assert isinstance(generator, dict)
    eligible_words = generator["eligibleWords"]
    assert isinstance(eligible_words, list)
    missing_words = sorted(set(eligible_words) - set(lexicon.token_to_pair))
    if missing_words:
        raise ValueError(
            "synthetic eligible words missing from lexicon: " + ", ".join(missing_words)
        )
    marked_documents = tuple(
        Document(
            document_id=document.document_id,
            text=encode_text(
                document.text,
                key=key,
                document_id=document.document_id,
                density_bps=density_bps,
                lexicon=lexicon,
            ).text,
        )
        for document in documents
    )
    marked_score = score_corpus(
        marked_documents,
        key=key,
        density_bps=density_bps,
        lexicon=lexicon,
    )
    unmarked_score = score_corpus(
        documents,
        key=key,
        density_bps=density_bps,
        lexicon=lexicon,
    )
    wrong_keys = run_wrong_key_controls(
        marked_documents,
        density_bps=density_bps,
        lexicon=lexicon,
        count=wrong_key_count,
        seed=wrong_key_seed,
    )

    config = {
        "contextWidth": DEFAULT_CONTEXT_WIDTH,
        "detectionAlphaExact": {
            "denominator": DETECTION_ALPHA.denominator,
            "numerator": DETECTION_ALPHA.numerator,
        },
        "densityBps": density_bps,
        "fixtureSha256": fixture_sha256,
        "keySha256": hashlib.sha256(key).hexdigest(),
        "lexiconSha256": lexicon.sha256,
        "minActivePositions": DEFAULT_MIN_ACTIVE_POSITIONS,
        "schemeVersion": SCHEME_VERSION,
        "textContractVersion": TEXT_CONTRACT_VERSION,
        "tokenizerVersion": TOKENIZER_VERSION,
        "wrongKeyCount": wrong_key_count,
        "wrongKeySeedSha256": hashlib.sha256(wrong_key_seed).hexdigest(),
    }
    checks = {
        "markedDetected": marked_score.status == "detected",
        "markedHasAtLeast100ActivePositions": marked_score.active_positions >= 100,
        "markedHitsEqualActivePositions": marked_score.hits == marked_score.active_positions,
        "unmarkedNotDetected": unmarked_score.status == "not_detected",
        "wrongKeyDetectionRateAtMost2_5Percent": (
            wrong_keys.detected_rate is not None and wrong_keys.detected_rate <= 0.025
        ),
        "wrongKeysAllSufficient": wrong_keys.insufficient_count == 0,
    }
    return {
        "acceptance": {"checks": checks, "passed": all(checks.values())},
        "config": config,
        "configSha256": _sha256_json(config),
        "fixtureVersion": fixture["fixtureVersion"],
        "markedCorpusSha256": _corpus_sha256(marked_documents),
        "methodology": fixture["methodology"],
        "originalCorpusSha256": _corpus_sha256(documents),
        "results": {
            "markedTrueKey": marked_score.to_dict(include_documents=True),
            "unmarkedTrueKey": unmarked_score.to_dict(include_documents=True),
            "wrongKeysOnMarked": wrong_keys.to_dict(include_scores=True),
        },
        "schemaVersion": 1,
        "sources": fixture["sources"],
        "verifiedAt": fixture["verifiedAt"],
    }


def build_synthetic_documents(fixture: dict[str, object]) -> tuple[Document, ...]:
    if fixture.get("schemaVersion") != 1:
        raise ValueError("unsupported synthetic fixture schemaVersion")
    generator = fixture.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("fixture.generator must be an object")
    document_count = generator.get("documentCount")
    repetitions = generator.get("repetitionsPerDocument")
    eligible_words = generator.get("eligibleWords")
    if not isinstance(document_count, int) or document_count <= 0:
        raise ValueError("generator.documentCount must be positive")
    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("generator.repetitionsPerDocument must be positive")
    if not isinstance(eligible_words, list) or not eligible_words:
        raise ValueError("generator.eligibleWords must be a non-empty list")
    if not all(
        isinstance(word, str) and word.isascii() and word.isalpha() and word.islower()
        for word in eligible_words
    ):
        raise ValueError("generator.eligibleWords must contain lowercase ASCII words")

    documents: list[Document] = []
    for document_index in range(document_count):
        fragments: list[str] = []
        for repetition in range(repetitions):
            for offset, word in enumerate(eligible_words):
                label = _alpha_label(repetition * len(eligible_words) + offset)
                fragments.append(f"context{label} {word}.")
        documents.append(
            Document(
                document_id=f"synthetic-{document_index:02d}",
                text=" ".join(fragments),
            )
        )
    return tuple(documents)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _corpus_sha256(documents: Sequence[Document]) -> str:
    return _sha256_json(
        [{"documentId": document.document_id, "text": document.text} for document in documents]
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _alpha_label(number: int) -> str:
    characters: list[str] = []
    value = number
    while True:
        value, remainder = divmod(value, 26)
        characters.append(chr(ord("a") + remainder))
        if value == 0:
            return "".join(reversed(characters))
        value -= 1


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        default=str(root / "fixtures" / "synthetic-preflight-v1.json"),
    )
    parser.add_argument(
        "--lexicon",
        default=str(root / "fixtures" / "synonym_pairs-v1.json"),
    )
    parser.add_argument("--key-hex", required=True)
    parser.add_argument("--density-bps", type=int, default=1_000)
    parser.add_argument("--wrong-key-count", type=int, default=1_000)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that --output already matches the regenerated artifact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    fixture_path = Path(args.fixture)
    fixture_bytes = fixture_path.read_bytes()
    fixture = json.loads(fixture_bytes)
    lexicon = load_lexicon(args.lexicon)
    try:
        key = bytes.fromhex(args.key_hex)
    except ValueError as error:
        raise SystemExit("--key-hex must be valid hexadecimal") from error
    result = run_preflight(
        fixture=fixture,
        fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
        lexicon=lexicon,
        key=key,
        density_bps=args.density_bps,
        wrong_key_count=args.wrong_key_count,
    )
    output_path = Path(args.output)
    artifact_bytes = canonical_json_bytes(result)
    artifact_matches = output_path.exists() and output_path.read_bytes() == artifact_bytes
    if not args.check:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(artifact_bytes)
        artifact_matches = True
    passed = result["acceptance"]["passed"] and artifact_matches
    print(
        json.dumps(
            {
                "artifactMatches": artifact_matches,
                "configSha256": result["configSha256"],
                "output": str(output_path),
                "passed": passed,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_WRONG_KEY_SEED",
    "build_synthetic_documents",
    "canonical_json_bytes",
    "run_preflight",
]
