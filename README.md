# Scientific Reasoning Audit Loops

## What I wanted to know

Can a small open-weight model improve its scientific problem solving by checking
its own work, and does a narrow deterministic verifier help more than an
unguided second look?

I built this project to connect my three research stacks. The science is a set
of chemistry, physics, and biology calculations. The AI work is a paired
inference-time intervention on Qwen3 1.7B. The security idea is the same one I
use in proof-carrying policy evaluation: do not trust a fluent answer when a
small deterministic check can test a specific claim.

I did not tune until the result looked good. I used 18 development cases to
repair the measurement interface, froze it, and then ran 72 untouched held-out
cases once. The repository keeps the generated scratchpads, choices, logits,
metadata, failures, and uncertainty estimates.

## The experiment

The dataset contains nine deterministic templates, with two development and
eight test cases per template:

| Domain | Templates |
| --- | --- |
| Chemistry | dilution, first-order half-life, ideal gas law |
| Physics | kinetic energy, Ohm's law, wavelength-to-frequency |
| Biology | Hardy–Weinberg carriers, Michaelis–Menten rate, population doubling |

Every case has four shuffled numerical choices. The generator computes the
correct value and constructs distractors from named failure modes such as a
missing unit conversion, a reversed ratio, or a missing exponent.

Qwen first generates a short scientific scratchpad. I then force a comparable
choice measurement by appending `FINAL:` and scoring the model's next-token
logits for the single-token labels A/B/C/D. This avoids mistaking a formatting
failure for a reasoning failure.

I compare three paired conditions:

1. **Direct:** score a choice from the initial scratchpad.
2. **Intrinsic audit:** ask the same model to recompute the key step and check
   formula and units, then score again.
3. **Verifier-gated:** if the direct choice is wrong, a deterministic checker
   supplies only the associated error category, never the correct answer. The
   model writes a corrected scratchpad and is rescored. Correct direct cases
   pass through unchanged.

## Held-out result

Model: `Qwen/Qwen3-1.7B` at revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, deterministic decoding, one RTX
4060 Laptop GPU, seed 101.

| Condition | Correct | Accuracy | Paired change vs. direct | 95% bootstrap CI | Wrong→right | Right→wrong | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct | 44/72 | 61.1% | — | — | — | — | — |
| Intrinsic audit | 57/72 | 79.2% | +18.1 points | +6.9 to +29.2 | 16 | 3 | 0.0044 |
| Verifier-gated | 55/72 | 76.4% | +15.3 points | +6.9 to +23.6 | 11 | 0 | 0.0010 |

The main result is not that more structure automatically wins. Unguided audit
had the highest total accuracy, while the verifier gate had the safer transition
profile because it never reopened initially correct cases. The two approaches
therefore optimize different things.

| Domain | Direct | Intrinsic audit | Verifier-gated |
| --- | ---: | ---: | ---: |
| Biology | 70.8% | 87.5% | 87.5% |
| Chemistry | 58.3% | 83.3% | 79.2% |
| Physics | 54.2% | 66.7% | 62.5% |

The worst direct template was wavelength-to-frequency at 12.5%. Intrinsic audit
raised it to 50.0%, but verifier feedback reached only 25.0%. Intrinsic audit
also reduced Hardy–Weinberg accuracy from 87.5% to 62.5%. Those failures show
why I store transition-level evidence instead of reporting only a mean.

## Public balanced-label confirmation

I ran the precommitted confirmation on a fresh 72-case cohort where every one
of the nine templates contains two correct A, B, C, and D options. The protocol,
generator, cases, manifest, tests, and evaluator revision were public and green
in CI before the single GPU run.

| Condition | Correct | Accuracy | Paired change vs. direct | 95% bootstrap CI | Wrong→right | Right→wrong | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct | 45/72 | 62.5% | — | — | — | — | — |
| Intrinsic audit | 54/72 | 75.0% | +12.5 points | +2.8 to +23.6 | 12 | 3 | 0.0352 |
| Verifier-gated | 59/72 | 81.9% | +19.4 points | +11.1 to +29.2 | 14 | 0 | 0.00012 |

Both conditions satisfy the frozen replication rule on the balanced cohort.
The verifier-gated condition again had the stronger transition profile because
it repaired 14 initially wrong cases while its gate left all initially correct
direct cases untouched. Intrinsic audit still improved accuracy but reopened and
regressed three correct direct cases.

This is a replication of an inference-time scaffold on a controlled numerical
benchmark, not evidence that Qwen acquired scientific knowledge. The verifier
still uses an oracle-derived error category after a wrong direct answer. The
important strengthened conclusion is narrower: the earlier effect was not
explained solely by an imbalanced A/B/C/D label distribution.

Run `python verify_balanced_confirmation.py` to validate the public protocol,
exactly balanced cases, output hashes, transition counts, and inferential
statistics.

## Same-family model-size replication: did not pass

I then ran the unchanged public balanced-label protocol once on the locally
cached Qwen3 0.6B. This is a model-size comparison inside the Qwen family, not
an independent-architecture replication.

| Condition | Correct | Accuracy | Paired change | 95% bootstrap CI | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 32/72 | 44.4% | — | — | — |
| Intrinsic audit | 35/72 | 48.6% | +4.2 points | 0.0 to +9.7 | 0.25 |
| Verifier-gated | 33/72 | 45.8% | +1.4 points | 0.0 to +4.2 | 1.0 |

Neither audit condition met the frozen replication rule: both lower confidence
bounds touched zero and neither McNemar test was significant. The direct model
also chose A on 41/72 cases despite exactly balanced correct labels. The 1.7B
result therefore does not automatically scale down to Qwen3 0.6B under this
pipeline. The result is a useful lower-capability boundary, not proof of a
general scale law or evidence that the smaller model cannot solve science.

Run `python verify_qwen_0_6b_balanced_replication.py` to verify the exact
frozen artifact and non-replication statistics.

## What this does and does not show

- It shows a statistically detectable paired improvement on this frozen
  synthetic set for this model and prompt loop.
- It does **not** show that the model learned new weights or became a generally
  better scientist. This is inference-time scaffolding.
- The verifier uses ground-truth-derived error categories and protects correct
  direct answers by design. Its zero regressions are therefore a property of
  the gate as well as the model.
- The test answer labels are not perfectly balanced (`A=14, B=26, C=19, D=13`);
  a majority-label baseline is 36.1%. Shuffled options and paired comparisons
  reduce but do not erase label-prior concerns.
- Some 96-token scratchpads end mid-calculation. A forced-choice label can also
  disagree with the visible scratchpad, so I treat the label as model behavior,
  not a faithful transcript of internal reasoning.
- These are controlled textbook-style calculations with exact oracles, not a
  benchmark of open-ended scientific discovery.
- This is one model, one seed, and one hardware condition. Replication across
  another architecture is the next serious test.

## Reproduce it

Using the existing CUDA environment from PowerShell:

```powershell
python .\science_cases.py
pytest -q
python .\run_evaluation.py --split test --output .\results\qwen3_1.7b_test.json
python .\verify_claims.py
```

The frozen dataset SHA-256 is
`5e8fc3a3b5713cbc973e68ee64d70d344549cbe4e92253be289c77a0835605b3`.
The result artifact SHA-256 is
`2c66207711a80da0bd79fe6bb71211935dfaefeb0fedc9d7a95290a79fd3db6b`.

The model run took 281.3 seconds after weights loaded. No paid API or cloud GPU
was used.

## Why this design is connected to existing research

- [SciBench](https://arxiv.org/abs/2307.10635) reports that scientific reasoning
  remains difficult and that prompting methods can help some skills while
  hurting others.
- The TACL survey [When Can LLMs Actually Correct Their Own
  Mistakes?](https://aclanthology.org/2024.tacl-1.78/) distinguishes unreliable
  intrinsic critique from correction supported by reliable external feedback.
- [LLMs cannot find reasoning errors, but can correct them given the error
  location](https://aclanthology.org/2024.findings-acl.826/) motivates separating
  mistake discovery from mistake repair.
- [ProCo](https://aclanthology.org/2024.emnlp-main.714/) shows that verifying a
  key condition can make self-correction more useful than a generic request to
  rethink an answer.

My benchmark is smaller and synthetic. Its value is that every case has an
executable oracle, every intervention is paired, and every negative transition
stays visible.

## Next experiments

1. Freeze a v2 dataset with exactly balanced answer labels and new numerical
   ranges before evaluating it.
2. Replicate on Gemma or OLMo so the result is not a Qwen-specific prompt effect.
3. Compare generic error categories with executable numeric residuals that do
   not disclose the answer.
4. Add uncertainty propagation and experimental-design questions, not only
   closed-form calculations.
5. Test whether an audit changes the hidden-state evidence representation before
   it changes the output label, connecting this repo to my evidence-conflict
   circuit work.

## Development record

The measurement failures that led to the final design are recorded in
[`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md). I include them because a
reproducible project should explain why the final metric exists, not pretend the
first implementation worked.
