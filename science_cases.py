"""Deterministic synthetic science problems with exact numerical oracles."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path


LETTERS = "ABCD"
TEMPLATES = (
    "chem_dilution",
    "chem_half_life",
    "chem_ideal_gas",
    "physics_kinetic_energy",
    "physics_ohms_law",
    "physics_frequency",
    "biology_hardy_weinberg",
    "biology_michaelis_menten",
    "biology_doubling",
)


def _fmt(value: float) -> str:
    if abs(value) >= 10_000 or (0 < abs(value) < 0.01):
        return f"{value:.3e}"
    return f"{value:.4g}"


def _pack(
    *, case_id: str, split: str, domain: str, template: str, question: str,
    unit: str, correct: float, distractors: list[tuple[float, str]], rng: random.Random,
) -> dict:
    candidates = [(correct, "correct")]
    for value, error in distractors:
        if math.isfinite(value) and all(not math.isclose(value, v, rel_tol=1e-7, abs_tol=1e-10) for v, _ in candidates):
            candidates.append((value, error))
    if len(candidates) != 4:
        raise ValueError(f"{case_id} did not produce four unique choices")
    rng.shuffle(candidates)
    options = {letter: f"{_fmt(value)} {unit}".strip() for letter, (value, _) in zip(LETTERS, candidates)}
    errors = {letter: error for letter, (_, error) in zip(LETTERS, candidates)}
    answer = next(letter for letter, error in errors.items() if error == "correct")
    return {
        "id": case_id,
        "split": split,
        "domain": domain,
        "template": template,
        "question": question,
        "options": options,
        "answer": answer,
        "error_by_choice": errors,
    }


def _case(template: str, index: int, split: str, rng: random.Random) -> dict:
    cid = f"{split}-{template}-{index:02d}"
    if template == "chem_dilution":
        c1 = rng.choice([20, 25, 40, 50, 80, 100])
        c2 = rng.choice([1, 2, 4, 5, 8, 10])
        if c2 >= c1:
            c2 = 1
        v2 = rng.choice([10, 20, 25, 40, 50])
        correct = c2 * v2 / c1
        return _pack(case_id=cid, split=split, domain="chemistry", template=template,
            question=f"A {c1} mM stock is diluted to make {v2} mL at {c2} mM. What stock volume is required?",
            unit="mL", correct=correct,
            distractors=[(c1 * v2 / c2, "recheck the dilution ratio"), (c2 * v2, "recheck the concentration division"), (correct * 1000, "recheck the volume-unit conversion")], rng=rng)
    if template == "chem_half_life":
        c0 = rng.choice([40, 64, 80, 96, 120, 160])
        half = rng.choice([2, 3, 4, 5, 6])
        periods = rng.choice([2, 3, 4])
        correct = c0 * 0.5**periods
        return _pack(case_id=cid, split=split, domain="chemistry", template=template,
            question=f"A first-order species starts at {c0} µM and has a {half} h half-life. What concentration remains after {half * periods} h?",
            unit="µM", correct=correct,
            distractors=[(c0 / periods, "recheck exponential versus linear decay"), (c0 * 0.5 * periods, "recheck how half-lives compound"), (c0 * 0.5 ** (periods + 1), "recheck the number of elapsed half-lives")], rng=rng)
    if template == "chem_ideal_gas":
        pressure = rng.choice([0.8, 1.0, 1.2, 1.5, 2.0])
        volume = rng.choice([1.5, 2.0, 3.0, 4.0, 5.0])
        temp_c = rng.choice([20, 25, 40, 60, 80])
        temp_k = temp_c + 273.15
        correct = pressure * volume / (0.082057 * temp_k)
        return _pack(case_id=cid, split=split, domain="chemistry", template=template,
            question=f"An ideal gas has P={pressure} atm, V={volume} L, and T={temp_c} °C. Using R=0.082057 L·atm·mol⁻¹·K⁻¹, how many moles are present?",
            unit="mol", correct=correct,
            distractors=[(pressure * volume / (0.082057 * temp_c), "convert Celsius to kelvin"), (pressure * volume * 0.082057 * temp_k, "recheck the algebra in PV=nRT"), (pressure * volume / 0.082057, "include absolute temperature")], rng=rng)
    if template == "physics_kinetic_energy":
        mass_g = rng.choice([50, 80, 120, 200, 250, 400])
        velocity = rng.choice([3, 4, 5, 6, 8, 10])
        correct = 0.5 * (mass_g / 1000) * velocity**2
        return _pack(case_id=cid, split=split, domain="physics", template=template,
            question=f"A {mass_g} g object moves at {velocity} m/s. What is its kinetic energy?",
            unit="J", correct=correct,
            distractors=[(0.5 * mass_g * velocity**2, "convert grams to kilograms"), ((mass_g / 1000) * velocity**2, "include the one-half factor"), (0.5 * (mass_g / 1000) * velocity, "square the velocity")], rng=rng)
    if template == "physics_ohms_law":
        voltage = rng.choice([3, 5, 6, 9, 12, 15])
        resistance = rng.choice([120, 150, 220, 330, 470, 680])
        correct = voltage / resistance * 1000
        return _pack(case_id=cid, split=split, domain="physics", template=template,
            question=f"A {voltage} V source is applied across a {resistance} Ω resistor. What current flows?",
            unit="mA", correct=correct,
            distractors=[(voltage / resistance, "convert amperes to milliamperes"), (voltage * resistance, "recheck Ohm's-law division"), (resistance / voltage, "recheck which quantity is divided")], rng=rng)
    if template == "physics_frequency":
        wavelength = rng.choice([400, 450, 500, 550, 600, 650, 700])
        correct = 3e5 / wavelength
        return _pack(case_id=cid, split=split, domain="physics", template=template,
            question=f"Light has wavelength {wavelength} nm. Using c=3.00×10^8 m/s, what is its frequency?",
            unit="THz", correct=correct,
            distractors=[(3e8 / wavelength, "convert nanometres and hertz consistently"), (3e2 / wavelength, "recheck the metric-prefix conversion"), (wavelength / 3e5, "recheck frequency versus wavelength")], rng=rng)
    if template == "biology_hardy_weinberg":
        affected_pct = rng.choice([1, 4, 9, 16, 36])
        q = math.sqrt(affected_pct / 100)
        correct = 2 * (1 - q) * q * 100
        return _pack(case_id=cid, split=split, domain="biology", template=template,
            question=f"In Hardy–Weinberg equilibrium, {affected_pct}% of a population has a recessive phenotype (q²). What percentage are heterozygous carriers (2pq)?",
            unit="%", correct=correct,
            distractors=[(affected_pct, "do not reuse the recessive phenotype frequency q²"), (q * 100, "compute 2pq rather than the allele frequency q"), ((1 - q) ** 2 * 100, "compute heterozygotes rather than homozygous dominant p²")], rng=rng)
    if template == "biology_michaelis_menten":
        vmax = rng.choice([60, 80, 100, 120, 150])
        km = rng.choice([2, 4, 5, 8, 10])
        substrate = rng.choice([value for value in [2, 4, 5, 8, 10, 16] if value != km])
        correct = vmax * substrate / (km + substrate)
        return _pack(case_id=cid, split=split, domain="biology", template=template,
            question=f"An enzyme has Vmax={vmax} µmol/min and Km={km} mM. At [S]={substrate} mM, what is v under Michaelis–Menten kinetics?",
            unit="µmol/min", correct=correct,
            distractors=[(vmax * substrate / km, "include Km+[S] in the denominator"), (vmax * km / (km + substrate), "place substrate, not Km, in the numerator"), (vmax / (km + substrate), "include the substrate factor")], rng=rng)
    if template == "biology_doubling":
        initial = rng.choice([2, 3, 4, 5, 8, 10])
        doubling = rng.choice([20, 30, 40, 60])
        periods = rng.choice([3, 4, 5])
        correct = initial * 2**periods
        return _pack(case_id=cid, split=split, domain="biology", template=template,
            question=f"A culture starts with {initial}×10^6 cells and doubles every {doubling} min. How many cells are present after {doubling * periods} min?",
            unit="×10^6 cells", correct=correct,
            distractors=[(initial + periods, "use exponential multiplication rather than adding once per period"), (initial * 2 ** (periods - 1), "recheck the number of doublings"), (initial**periods, "use a base of two for doubling")], rng=rng)
    raise KeyError(template)


def generate(seed: int = 240827, dev_per_template: int = 2, test_per_template: int = 8) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    for template in TEMPLATES:
        for split, count in (("dev", dev_per_template), ("test", test_per_template)):
            for index in range(count):
                cases.append(_case(template, index, split, rng))
    return cases


def write_dataset(path: Path) -> str:
    payload = {"schema_version": 1, "seed": 240827, "cases": generate()}
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    target = Path(__file__).parent / "data" / "science_cases_v1.json"
    digest = write_dataset(target)
    print(f"wrote {target} sha256={digest}")
