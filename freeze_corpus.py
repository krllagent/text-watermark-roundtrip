"""Validate the key-free article corpus and write deterministic freeze artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from corpus_contract import (
    build_context_inventory,
    build_manifest,
    canonical_json_bytes,
    inspect_corpus,
    load_corpus_plan,
    validate_context_reviews,
)
from watermark_toy import load_lexicon


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(root))
    parser.add_argument("--plan", default=str(root / "fixtures" / "corpus-plan-v1.json"))
    parser.add_argument(
        "--lexicon",
        default=str(root / "fixtures" / "synonym_pairs-v1.json"),
    )
    parser.add_argument(
        "--manifest",
        default=str(root / "corpus" / "manifest-v1.json"),
    )
    parser.add_argument(
        "--inventory",
        default=str(root / "corpus" / "context-inventory-v1.json"),
    )
    parser.add_argument(
        "--reviews-dir",
        default=str(root / "corpus" / "reviews"),
    )
    parser.add_argument("--require-approved", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    plan_path = Path(args.plan)
    plan_bytes = plan_path.read_bytes()
    plan = load_corpus_plan(plan_path)
    lexicon = load_lexicon(args.lexicon)
    documents = inspect_corpus(args.root, plan=plan, lexicon=lexicon)
    manifest = build_manifest(
        plan=plan,
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        lexicon=lexicon,
        documents=documents,
    )
    inventory = build_context_inventory(plan=plan, lexicon=lexicon, documents=documents)
    approval: dict[str, object] | None = None
    if args.require_approved:
        review_paths = sorted(Path(args.reviews_dir).glob("*.json"))
        reviews = [json.loads(path.read_text(encoding="utf-8")) for path in review_paths]
        approval = validate_context_reviews(inventory=inventory, reviews=reviews)
    outputs = {
        Path(args.manifest): canonical_json_bytes(manifest),
        Path(args.inventory): canonical_json_bytes(inventory),
    }
    matches = {
        str(path): path.exists() and path.read_bytes() == content
        for path, content in outputs.items()
    }
    if not args.check:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        matches = {str(path): True for path in outputs}
    passed = all(matches.values())
    print(
        json.dumps(
            {
                "documentCount": len(documents),
                "eligiblePositions": manifest["eligiblePositions"],
                "matches": matches,
                "passed": passed,
                "reviewApproval": approval,
                "wordCount": manifest["wordCount"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
