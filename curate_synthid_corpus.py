"""Apply an auditable minimal-edit manifest to a SynthID corpus.

The raw GPU artifact remains immutable.  This derivation validates that every
old phrase occurs exactly once, applies only the declared replacements, checks
the pinned tokenizer against every raw detector trace, and recomputes all
model-free SynthID detector fields for the edited text.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import statistics

from gen_quality_synthid_corpus_gpu import (
    WORD_PATTERN,
    detector_trace,
    quality_issues,
    utc_now,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "quality-synthid-corpus-gpu-v1.json"
DEFAULT_MANIFEST = ROOT / "configs" / "quality-synthid-curation-v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "quality-synthid-corpus-curated-v1.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_exact_edits(
    text: str, edits: list[dict[str, object]]
) -> tuple[str, list[str]]:
    edit_ids: list[str] = []
    for edit in edits:
        old = str(edit["old"])
        new = str(edit["new"])
        edit_id = str(edit["editId"])
        count = text.count(old)
        if count != 1:
            raise ValueError(
                f"curation edit {edit_id} old text must occur exactly once; found {count}"
            )
        if old == new:
            raise ValueError(f"curation edit {edit_id} does not change text")
        text = text.replace(old, new, 1)
        edit_ids.append(edit_id)
    return text, edit_ids


def curate(
    raw: dict[str, object],
    manifest: dict[str, object],
    *,
    raw_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    import torch
    from transformers import AutoTokenizer, SynthIDTextWatermarkLogitsProcessor

    edits = manifest.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("curation manifest must contain edits")
    edit_ids = [str(edit["editId"]) for edit in edits]
    if len(edit_ids) != len(set(edit_ids)):
        raise ValueError("curation edit IDs must be unique")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for edit in edits:
        side = str(edit["side"])
        if side not in ("marked", "unmarked"):
            raise ValueError(f"invalid curation side: {side}")
        grouped[(str(edit["documentId"]), side)].append(edit)

    tokenizer = AutoTokenizer.from_pretrained(
        str(raw["model"]), revision=str(raw["modelRevision"])
    )
    processor = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=int(raw["ngramLen"]),
        keys=raw["keys"],
        sampling_table_size=len(raw["samplingTable"]),
        sampling_table_seed=0,
        context_history_size=int(raw["contextHistorySize"]),
        device=torch.device("cpu"),
    )
    processor.sampling_table.copy_(
        torch.tensor(raw["samplingTable"], dtype=processor.sampling_table.dtype)
    )

    output = copy.deepcopy(raw)
    output["sourceCreatedAt"] = output.pop("createdAt")
    output["createdAt"] = utc_now()
    output["derivedFrom"] = {
        "path": str(raw_path.relative_to(ROOT)),
        "sha256": sha256_file(raw_path),
    }
    output["curation"] = {
        "editCount": len(edits),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifestSha256": sha256_file(manifest_path),
        "methodology": manifest["methodology"],
    }
    output["methodology"] = (
        str(raw["methodology"])
        + " The accepted GPU outputs were then minimally curated by an exact-replacement "
        "manifest limited to unambiguous internal contradictions or malformed referents. "
        "No text was regenerated. The pinned tokenizer and serialized sampling table "
        "were used to recompute every detector trace after editing."
    )

    applied: list[str] = []
    known_documents = {str(document["documentId"]) for document in output["documents"]}
    unknown_documents = {key[0] for key in grouped} - known_documents
    if unknown_documents:
        raise ValueError(f"manifest references unknown documents: {unknown_documents}")

    for document in output["documents"]:
        document_id = str(document["documentId"])
        document["sourceGenerationAttempts"] = document.pop("attempts")
        for side in ("marked", "unmarked"):
            row = document[side]
            raw_text = str(row["text"])
            raw_ids = tokenizer(
                [raw_text], return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].tolist()
            if raw_ids != row["detectorTokenIds"]:
                raise ValueError(
                    f"pinned tokenizer does not reproduce raw trace for {document_id} {side}"
                )
            edited_text, row_edit_ids = apply_exact_edits(
                raw_text, grouped.get((document_id, side), [])
            )
            applied.extend(row_edit_ids)
            pre_curation = {
                "generationTokenIdsSha256": sha256_json(row["generationTokenIds"]),
                "meanG": row["meanG"],
                "stoppedOnEos": row["stoppedOnEos"],
                "textSha256": row["textSha256"],
                "tokenCount": row["tokenCount"],
                "validGValueCount": row["validGValueCount"],
            }
            for key in (
                "contextRepetitionMask",
                "decodeRetokenizationExact",
                "detectorTokenIds",
                "gValues",
                "generationTokenIds",
                "meanG",
                "perDepthMeanG",
                "qualityIssues",
                "stopTokenId",
                "stoppedOnEos",
                "textSha256",
                "tokenCount",
                "validGValueCount",
                "wordCount",
            ):
                row.pop(key, None)
            row["text"] = edited_text
            trace = detector_trace(processor, tokenizer, edited_text)
            row.update(trace)
            row["tokenCount"] = len(trace["detectorTokenIds"])
            row["wordCount"] = len(WORD_PATTERN.findall(edited_text))
            row["textSha256"] = hashlib.sha256(edited_text.encode("utf-8")).hexdigest()
            row["qualityIssues"] = quality_issues(
                edited_text,
                token_count=row["tokenCount"],
                max_new_tokens=int(raw["generation"]["maxNewTokens"]),
                stopped_on_eos=bool(pre_curation["stoppedOnEos"]),
            )
            if row["qualityIssues"]:
                raise ValueError(
                    f"curated quality contract failed for {document_id} {side}: "
                    f"{row['qualityIssues']}"
                )
            row["curated"] = bool(row_edit_ids)
            row["curationEditIds"] = row_edit_ids
            row["preCuration"] = pre_curation
            row["scoreDeltaFromPreCuration"] = (
                float(row["meanG"]) - float(pre_curation["meanG"])
            )

    if sorted(applied) != sorted(edit_ids):
        raise ValueError("not every declared curation edit was applied exactly once")
    output["markedMeanG"] = statistics.mean(
        float(document["marked"]["meanG"]) for document in output["documents"]
    )
    output["unmarkedMeanG"] = statistics.mean(
        float(document["unmarked"]["meanG"]) for document in output["documents"]
    )
    output["status"] = "complete"
    output["verifiedAt"] = utc_now()
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = curate(
        raw,
        manifest,
        raw_path=args.input,
        manifest_path=args.manifest,
    )
    write_json_atomic(args.output, output)
    print(
        json.dumps(
            {
                "editCount": output["curation"]["editCount"],
                "markedMeanG": output["markedMeanG"],
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "unmarkedMeanG": output["unmarkedMeanG"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
