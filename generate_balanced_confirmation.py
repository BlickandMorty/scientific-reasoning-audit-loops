"""Create a fresh exact-label-balanced science-audit confirmation cohort."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from science_cases import LETTERS, TEMPLATES, _case


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "docs" / "BALANCED_LABEL_CONFIRMATION_PREREGISTRATION.md"
SEED = 280828


def generate() -> list[dict]:
    cases = []
    for template_index, template in enumerate(TEMPLATES):
        targets = [letter for letter in LETTERS for _ in range(2)]
        for index, target in enumerate(targets):
            for attempt in range(10000):
                rng = random.Random(f"{SEED}:{template_index}:{index}:{attempt}")
                try:
                    candidate = _case(template, index, "confirm", rng)
                except ValueError:
                    # A few numeric combinations make two named distractors
                    # coincide. They are invalid candidates, never relabeled.
                    continue
                if candidate["answer"] == target:
                    candidate["id"] = f"confirm-{template}-{index:02d}"
                    cases.append(candidate)
                    break
            else:
                raise RuntimeError(f"could not produce {template} choice {target}")
    return cases


def main() -> None:
    cases = generate()
    answers = {letter: sum(case["answer"] == letter for case in cases) for letter in LETTERS}
    per_template = {template: {letter: sum(case["template"] == template and case["answer"] == letter for case in cases) for letter in LETTERS} for template in TEMPLATES}
    if len(cases) != 72 or answers != {letter: 18 for letter in LETTERS} or any(counts != {letter: 2 for letter in LETTERS} for counts in per_template.values()):
        raise RuntimeError("balanced-label cohort invariant failed")
    payload = {"schema_version": 1, "seed": SEED, "split": "confirm", "cases": cases}
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    data = ROOT / "data" / "science_cases_balanced_confirmation_v1.json"
    data.write_bytes(encoded)
    manifest = {"dataset": "science-audit-balanced-confirmation-v1", "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), "dataset_sha256": hashlib.sha256(encoded).hexdigest(), "seed": SEED, "cases": len(cases), "answer_counts": answers, "per_template_answer_counts": per_template, "domains": sorted({case["domain"] for case in cases}), "safety": "synthetic closed-form science calculations with exact numerical oracles; not experimental or clinical advice"}
    (ROOT / "data" / "science_cases_balanced_confirmation_manifest_v1.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
