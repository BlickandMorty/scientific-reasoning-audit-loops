# Publicly Frozen SmolLM2 Balanced-Label Replication

## Question

The balanced-label confirmation established the audit-loop effect on Qwen3
1.7B. This replication asks whether the same frozen inference-time pipeline
works on an independent public, ungated instruction model:
`HuggingFaceTB/SmolLM2-1.7B-Instruct`.

## Pre-inference lock

Before model inference, this protocol is committed publicly. Use the already
public balanced-label dataset unchanged, its 72 cases, `confirm` split, the
existing direct/intrinsic/verifier pipeline, CUDA float16, seed 101, batch size
8, scratchpad cap 96, forced A/B/C/D scoring, and one full pass. Downloading the
public model is permitted only through the normal Hugging Face cache; no paid
API, cloud job, model tuning, prompt change, dataset change, or repeat run is
permitted after outputs exist.

## Measurements and rule

Report direct accuracy, each paired audit change, 10,000-draw bootstrap 95%
interval, wrong-to-right/right-to-wrong transitions, exact McNemar p values,
and domain/template slices. An effect replicates only if accuracy increases,
the paired lower interval is strictly positive, and exact McNemar p < .05.
Verifier zero regressions remain gate behavior, not evidence that the model has
learned scientific knowledge.

Any outcome concerns a small controlled numerical benchmark and inference-time
scaffolding. It does not establish general scientific reasoning or a universal
cross-model effect.
