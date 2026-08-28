import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from science_cases import LETTERS, TEMPLATES, generate
from generate_balanced_confirmation import generate as generate_balanced


def test_dataset_is_balanced_and_unique():
    cases = generate()
    assert len(cases) == 90
    assert len({c["id"] for c in cases}) == 90
    for template in TEMPLATES:
        assert sum(c["template"] == template and c["split"] == "test" for c in cases) == 8


def test_oracle_and_diagnostics_cover_every_choice():
    for case in generate():
        assert set(case["options"]) == set(LETTERS)
        assert set(case["error_by_choice"]) == set(LETTERS)
        assert case["error_by_choice"][case["answer"]] == "correct"
        assert len(set(case["options"].values())) == 4


def test_generation_is_deterministic():
    assert generate() == generate()


def test_balanced_confirmation_has_equal_answer_labels_per_template():
    cases = generate_balanced()
    assert len(cases) == 72
    assert {letter: sum(case["answer"] == letter for case in cases) for letter in LETTERS} == {"A": 18, "B": 18, "C": 18, "D": 18}
    for template in TEMPLATES:
        rows = [case for case in cases if case["template"] == template]
        assert len(rows) == 8
        assert {letter: sum(case["answer"] == letter for case in rows) for letter in LETTERS} == {"A": 2, "B": 2, "C": 2, "D": 2}
