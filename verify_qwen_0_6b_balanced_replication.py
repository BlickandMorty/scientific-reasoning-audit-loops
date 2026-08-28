"""Verify the Qwen3 0.6B same-family model-size replication artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from generate_balanced_confirmation import generate


ROOT = Path(__file__).resolve().parent
EXPECTED = {"protocol": "88f31e39505d63bbdaea990bcd2e68f8e7c2b11ca8b210921e9d9a32c2d78b16", "data": "d7e5a7aa0864123a3baf42fc926513ce0a6e715909cbb84a1806d24f55bcaace", "manifest": "ecfb206f73fa8defbaeabc58150cd2c035b8430190c3ef56518b9f1ae5987c82", "result": "66c261cec8dde7f4a013053f0f894165efb00883f8562cbc876ea7aebe1a305b"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    paths = {"protocol": ROOT / "docs" / "QWEN_0_6B_BALANCED_REPLICATION_PREREGISTRATION.md", "data": ROOT / "data" / "science_cases_balanced_confirmation_v1.json", "manifest": ROOT / "data" / "science_cases_balanced_confirmation_manifest_v1.json", "result": ROOT / "results" / "qwen3_0.6b_balanced_confirmation_v1.json"}
    for name, path in paths.items():
        assert digest(path) == EXPECTED[name], f"{name} hash changed"
    assert json.loads(paths["data"].read_text(encoding="utf-8"))["cases"] == generate()
    report = json.loads(paths["result"].read_text(encoding="utf-8"))
    rows = report["rows"]
    assert len(rows) == 72
    assert (sum(row["direct_correct"] for row in rows), sum(row["intrinsic_correct"] for row in rows), sum(row["verifier_correct"] for row in rows)) == (32, 35, 33)
    summary = report["summary"]
    assert summary["intrinsic_audit"]["bootstrap_95_ci_for_change"] == [0.0, 0.09722222222222222]
    assert summary["intrinsic_audit"]["mcnemar_exact_two_sided_p"] == 0.25
    assert summary["verifier_gated"]["bootstrap_95_ci_for_change"] == [0.0, 0.041666666666666664]
    assert summary["verifier_gated"]["mcnemar_exact_two_sided_p"] == 1.0
    assert summary["choice_counts"]["direct"] == {"A": 41, "B": 13, "C": 7, "D": 11}
    print("verified: Qwen3 0.6B did not replicate the balanced audit-loop effect")


if __name__ == "__main__":
    main()
