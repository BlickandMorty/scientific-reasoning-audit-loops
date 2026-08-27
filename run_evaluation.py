"""Compare direct, intrinsic-audit, and verifier-gated science reasoning."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer


LETTERS = "ABCD"


def format_problem(case: dict) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in case["options"].items())
    return f"{case['question']}\n\n{options}"


def solution_message(case: dict) -> str:
    return (
        "Create a compact scratchpad for this scientific problem. State the governing formula, substitute the values, "
        "and check units. Do not choose a letter and use at most 60 words.\n\n"
        + format_problem(case)
    )


def choice_message(case: dict, scratchpad: str) -> str:
    return (
        "Use the problem and scratchpad below. Return only the best choice after FINAL:.\n\n"
        f"PROBLEM:\n{format_problem(case)}\n\nSCRATCHPAD:\n{scratchpad}"
    )


def intrinsic_audit_message(case: dict, scratchpad: str, choice: str) -> str:
    return (
        "Audit this proposed scientific solution without assuming it is right or wrong. Independently recompute the key step, "
        "check the formula and units, and state whether the proposed choice should change. Do not output a final letter; use at most 60 words.\n\n"
        f"PROBLEM:\n{format_problem(case)}\n\nINITIAL SCRATCHPAD:\n{scratchpad}\n\nPROPOSED CHOICE: {choice}"
    )


def verifier_audit_message(case: dict, scratchpad: str, choice: str, feedback: str) -> str:
    return (
        "A deterministic checker rejected the proposed choice and named an error category without revealing the answer. "
        f"FEEDBACK: {feedback}. Recompute the relevant step and write a corrected scratchpad. Do not output a final letter; use at most 60 words.\n\n"
        f"PROBLEM:\n{format_problem(case)}\n\nINITIAL SCRATCHPAD:\n{scratchpad}\n\nPROPOSED CHOICE: {choice}"
    )


def audited_choice_message(case: dict, initial: str, audit: str) -> str:
    return (
        "Use the problem, initial scratchpad, and audit below. Return only the best choice after FINAL:.\n\n"
        f"PROBLEM:\n{format_problem(case)}\n\nINITIAL SCRATCHPAD:\n{initial}\n\nAUDIT:\n{audit}"
    )


def chat_prompt(tokenizer, message: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def score_batch(model, tokenizer, messages: list[str], batch_size: int) -> tuple[list[str], list[dict]]:
    choices = []
    score_rows = []
    candidate_ids = []
    for letter in LETTERS:
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Candidate {letter} is not one token: {ids}")
        candidate_ids.append(ids[0])
    for start in range(0, len(messages), batch_size):
        prompts = [chat_prompt(tokenizer, m) + "FINAL:" for m in messages[start:start + batch_size]]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, candidate_ids].float().cpu().numpy()
        for row in logits:
            choices.append(LETTERS[int(np.argmax(row))])
            score_rows.append({letter: float(value) for letter, value in zip(LETTERS, row)})
    return choices, score_rows


def generate_batch(model, tokenizer, messages: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    outputs = []
    for start in range(0, len(messages), batch_size):
        prompts = [chat_prompt(tokenizer, m) for m in messages[start:start + batch_size]]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outputs


def bootstrap_diff(a: list[int], b: list[int], seed: int = 9182, draws: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    delta = np.asarray(b) - np.asarray(a)
    means = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(draws)])
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def paired_summary(rows: list[dict], arm: str) -> dict:
    before = [int(r["direct_correct"]) for r in rows]
    after = [int(r[f"{arm}_correct"]) for r in rows]
    fixed = sum(x == 0 and y == 1 for x, y in zip(before, after))
    broken = sum(x == 1 and y == 0 for x, y in zip(before, after))
    discordant = fixed + broken
    pvalue = 1.0 if discordant == 0 else float(binomtest(min(fixed, broken), discordant, 0.5).pvalue)
    return {
        "accuracy": sum(after) / len(after),
        "paired_accuracy_change": (sum(after) - sum(before)) / len(after),
        "bootstrap_95_ci_for_change": bootstrap_diff(before, after),
        "wrong_to_right": fixed,
        "right_to_wrong": broken,
        "mcnemar_exact_two_sided_p": pvalue,
    }


def summarize(rows: list[dict]) -> dict:
    summary = {
        "n": len(rows),
        "direct_accuracy": sum(r["direct_correct"] for r in rows) / len(rows),
        "intrinsic_audit": paired_summary(rows, "intrinsic"),
        "verifier_gated": paired_summary(rows, "verifier"),
        "choice_counts": {arm: dict(Counter(r[f"{arm}_choice"] for r in rows)) for arm in ("direct", "intrinsic", "verifier")},
    }
    for slice_name in ("domain", "template"):
        buckets = defaultdict(list)
        for row in rows:
            buckets[row[slice_name]].append(row)
        summary[f"by_{slice_name}"] = {
            key: {
                "n": len(items),
                "direct": sum(r["direct_correct"] for r in items) / len(items),
                "intrinsic": sum(r["intrinsic_correct"] for r in items) / len(items),
                "verifier": sum(r["verifier_correct"] for r in items) / len(items),
            }
            for key, items in sorted(buckets.items())
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=Path, default=Path("data/science_cases_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/qwen3_1.7b_test.json"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scratchpad-tokens", type=int, default=96)
    args = parser.parse_args()

    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    raw_dataset = args.dataset.read_bytes()
    data = json.loads(raw_dataset)
    cases = [c for c in data["cases"] if c["split"] == args.split]
    if args.limit:
        cases = cases[:args.limit]

    if not torch.cuda.is_available():
        raise RuntimeError("This frozen run expects CUDA; use a smaller smoke test or record a new hardware condition.")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()

    started = time.time()
    initial_scratchpads = generate_batch(model, tokenizer, [solution_message(c) for c in cases], args.batch_size, args.scratchpad_tokens)
    direct_choices, direct_scores = score_batch(model, tokenizer, [choice_message(c, s) for c, s in zip(cases, initial_scratchpads)], args.batch_size)
    intrinsic_audits = generate_batch(
        model, tokenizer,
        [intrinsic_audit_message(c, s, choice) for c, s, choice in zip(cases, initial_scratchpads, direct_choices)],
        args.batch_size, args.scratchpad_tokens,
    )
    intrinsic_choices, intrinsic_scores = score_batch(
        model, tokenizer,
        [audited_choice_message(c, s, audit) for c, s, audit in zip(cases, initial_scratchpads, intrinsic_audits)],
        args.batch_size,
    )

    verifier_choices = list(direct_choices)
    verifier_scores = list(direct_scores)
    wrong_indices = [i for i, (c, choice) in enumerate(zip(cases, direct_choices)) if choice != c["answer"]]
    verifier_prompts = []
    for i in wrong_indices:
        choice = direct_choices[i]
        feedback = cases[i]["error_by_choice"][choice]
        verifier_prompts.append(verifier_audit_message(cases[i], initial_scratchpads[i], choice, feedback))
    verifier_audits = [None] * len(cases)
    if verifier_prompts:
        corrected_audits = generate_batch(model, tokenizer, verifier_prompts, args.batch_size, args.scratchpad_tokens)
        correction_prompts = [audited_choice_message(cases[i], initial_scratchpads[i], audit) for i, audit in zip(wrong_indices, corrected_audits)]
        corrected_choices, corrected_scores = score_batch(model, tokenizer, correction_prompts, args.batch_size)
        for i, choice, scores, audit in zip(wrong_indices, corrected_choices, corrected_scores, corrected_audits):
            verifier_choices[i] = choice
            verifier_scores[i] = scores
            verifier_audits[i] = audit

    rows = []
    for case, d, i, v, d_scores, i_scores, v_scores, scratch, intrinsic_audit, verifier_audit in zip(cases, direct_choices, intrinsic_choices, verifier_choices, direct_scores, intrinsic_scores, verifier_scores, initial_scratchpads, intrinsic_audits, verifier_audits):
        rows.append({
            "id": case["id"], "domain": case["domain"], "template": case["template"], "answer": case["answer"],
            "direct_choice": d, "intrinsic_choice": i, "verifier_choice": v,
            "direct_correct": d == case["answer"], "intrinsic_correct": i == case["answer"], "verifier_correct": v == case["answer"],
            "verifier_feedback": None if d == case["answer"] else case["error_by_choice"][d],
            "initial_scratchpad": scratch, "intrinsic_audit": intrinsic_audit, "verifier_audit": verifier_audit,
            "direct_label_logits": d_scores, "intrinsic_label_logits": i_scores, "verifier_label_logits": v_scores,
        })

    revision = getattr(model.config, "_commit_hash", None)
    report = {
        "schema_version": 1,
        "design": "frozen synthetic test set; generated scientific scratchpads followed by forced-choice label scoring; one intrinsic audit; verifier revises only failed cases and never reveals the answer",
        "model": args.model,
        "model_revision": revision,
        "dataset_sha256": hashlib.sha256(raw_dataset).hexdigest(),
        "seed": 101,
        "hardware": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "runtime_seconds": time.time() - started,
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
