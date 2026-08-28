"""Verify the public balanced-label confirmation and its exact result."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from generate_balanced_confirmation import generate
from science_cases import LETTERS, TEMPLATES


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "protocol": "4f029148d769e5387091afe6a1da76b706e6b95c596530f20d2fcb42ddf7c49b",
    "data": "d7e5a7aa0864123a3baf42fc926513ce0a6e715909cbb84a1806d24f55bcaace",
    "manifest": "ecfb206f73fa8defbaeabc58150cd2c035b8430190c3ef56518b9f1ae5987c82",
    "result": "59f063cc05c7868693902714d65b0d768192fbc85d962d1575810f95bd7092b0",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    paths = {
        "protocol": ROOT / "docs" / "BALANCED_LABEL_CONFIRMATION_PREREGISTRATION.md",
        "data": ROOT / "data" / "science_cases_balanced_confirmation_v1.json",
        "manifest": ROOT / "data" / "science_cases_balanced_confirmation_manifest_v1.json",
        "result": ROOT / "results" / "qwen3_1.7b_balanced_confirmation_v1.json",
    }
    for name, path in paths.items():
        assert digest(path) == EXPECTED[name], f"{name} hash changed: {digest(path)}"
    data = json.loads(paths["data"].read_text(encoding="utf-8"))
    assert data["cases"] == generate()
    cases = data["cases"]
    assert len(cases) == 72
    assert {letter: sum(case["answer"] == letter for case in cases) for letter in LETTERS} == {"A": 18, "B": 18, "C": 18, "D": 18}
    for template in TEMPLATES:
        assert {letter: sum(case["template"] == template and case["answer"] == letter for case in cases) for letter in LETTERS} == {"A": 2, "B": 2, "C": 2, "D": 2}
    report = json.loads(paths["result"].read_text(encoding="utf-8"))
    rows = report["rows"]
    assert len(rows) == 72
    assert report["dataset_sha256"] == "d7e5a7aa0864123a3baf42fc926513ce0a6e715909cbb84a1806d24f55bcaace"
    for row in rows:
        assert row["direct_correct"] == (row["direct_choice"] == row["answer"])
        assert row["intrinsic_correct"] == (row["intrinsic_choice"] == row["answer"])
        assert row["verifier_correct"] == (row["verifier_choice"] == row["answer"])
    summary = report["summary"]
    assert (sum(row["direct_correct"] for row in rows), sum(row["intrinsic_correct"] for row in rows), sum(row["verifier_correct"] for row in rows)) == (45, 54, 59)
    assert math.isclose(summary["intrinsic_audit"]["paired_accuracy_change"], 0.125)
    assert summary["intrinsic_audit"]["bootstrap_95_ci_for_change"] == [0.027777777777777776, 0.2361111111111111]
    assert summary["intrinsic_audit"]["mcnemar_exact_two_sided_p"] == 0.03515625
    assert summary["verifier_gated"]["wrong_to_right"] == 14
    assert summary["verifier_gated"]["right_to_wrong"] == 0
    assert summary["verifier_gated"]["bootstrap_95_ci_for_change"] == [0.1111111111111111, 0.2916666666666667]
    assert summary["verifier_gated"]["mcnemar_exact_two_sided_p"] == 0.0001220703125
    print("verified: public balanced-label science-audit confirmation")


if __name__ == "__main__":
    main()
