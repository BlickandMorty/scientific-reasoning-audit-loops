"""Verify frozen dataset and any committed result artifact."""

import hashlib
import json
import math
from pathlib import Path

from science_cases import generate


root = Path(__file__).parent
dataset_path = root / "data" / "science_cases_v1.json"
data = json.loads(dataset_path.read_text(encoding="utf-8"))
assert data["cases"] == generate()
assert len([c for c in data["cases"] if c["split"] == "dev"]) == 18
assert len([c for c in data["cases"] if c["split"] == "test"]) == 72
assert {c["domain"] for c in data["cases"]} == {"chemistry", "physics", "biology"}

result_path = root / "results" / "qwen3_1.7b_test.json"
if result_path.exists():
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["dataset_sha256"] == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert report["summary"]["n"] == 72
    assert len(report["rows"]) == 72
    for row in report["rows"]:
        assert row["direct_correct"] == (row["direct_choice"] == row["answer"])
        assert row["intrinsic_correct"] == (row["intrinsic_choice"] == row["answer"])
        assert row["verifier_correct"] == (row["verifier_choice"] == row["answer"])
    rows = report["rows"]
    direct = sum(row["direct_correct"] for row in rows)
    intrinsic = sum(row["intrinsic_correct"] for row in rows)
    verifier = sum(row["verifier_correct"] for row in rows)
    assert (direct, intrinsic, verifier) == (44, 57, 55)
    assert math.isclose(report["summary"]["direct_accuracy"], direct / 72)
    assert math.isclose(report["summary"]["intrinsic_audit"]["accuracy"], intrinsic / 72)
    assert math.isclose(report["summary"]["verifier_gated"]["accuracy"], verifier / 72)
    assert report["summary"]["intrinsic_audit"]["wrong_to_right"] == 16
    assert report["summary"]["intrinsic_audit"]["right_to_wrong"] == 3
    assert report["summary"]["verifier_gated"]["wrong_to_right"] == 11
    assert report["summary"]["verifier_gated"]["right_to_wrong"] == 0

print("scientific reasoning artifacts verified")
