# Publicly Frozen Balanced-Label Confirmation Protocol

## Question

The original 72-case Qwen3 scientific-audit result improved after intrinsic
audit and verifier-gated correction, but its answer labels were not perfectly
balanced. This confirmation asks whether those paired effects survive a fresh
chemistry, physics, and biology cohort with exactly balanced A/B/C/D labels.

## Pre-inference lock

Before model inference, this protocol, deterministic generator, generated
dataset, tests, and evaluator revision are committed publicly. The confirmation
uses `Qwen/Qwen3-1.7B`, its resolved revision, CUDA float16, seed 101, batch
size 8, and the existing frozen direct/intrinsic/verifier prompt pipeline. Run
the full cohort once. No retraining, prompt tuning, sampling, case replacement,
or metric change is allowed after output exists.

## Frozen cohort

Generate eight new cases for each of the existing nine templates (72 total).
Each template contains exactly two cases whose correct option is each of A, B,
C, and D, yielding 18 correct answers per label overall. The formulas,
numerical ranges, named distractor categories, exact numerical oracle, and
three domains remain the same as the original controlled textbook-style task;
the new cases use a distinct deterministic generator seed and `confirm` split.

## Conditions and measurements

Compare the same three paired conditions: direct forced-choice score after a
bounded scratchpad; intrinsic audit and rescore; verifier-gated audit only when
the direct answer is wrong, receiving its named error category but never the
answer. Primary measurements are paired accuracy changes and 10,000-draw
bootstrap 95% intervals for intrinsic and verifier conditions versus direct.
Secondary measurements are wrong-to-right, right-to-wrong, exact McNemar tests,
domain/template slices, and label-choice distributions.

## Decision rule

An original effect replicates only if its audit condition has higher accuracy
than direct, a strictly positive paired bootstrap lower bound, and an exact
McNemar p below .05. Verifier-gated zero regressions are reported separately,
not treated as evidence of model reasoning because the gate protects direct
correct cases by design. A pass concerns this frozen small-model benchmark, not
open-ended scientific discovery or learned scientific knowledge.
