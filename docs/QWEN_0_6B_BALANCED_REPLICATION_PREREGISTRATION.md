# Publicly Frozen Qwen3 0.6B Balanced-Label Replication

## Question

The Qwen3 1.7B balanced-label confirmation supports an audit-loop effect at one
model size. This test asks whether that effect persists in the locally cached
Qwen3 0.6B model under the same frozen benchmark and inference-time pipeline.

This is a **same-family model-size replication**, not an independent-architecture
replication. The separate SmolLM2 protocol remains unrun because its public
weight transfer stalled before model load.

## Pre-inference lock

Before inference, this protocol is publicly committed. Use the existing public
72-case balanced confirmation dataset, `confirm` split, Qwen/Qwen3-0.6B, its
resolved revision, CUDA float16, seed 101, batch size 8, scratchpad cap 96, and
the unchanged direct/intrinsic/verifier pipeline. Run exactly once. No tuning,
prompt change, data change, or second run after output exists.

## Measurements and rule

Report direct and paired audit accuracy, 10,000-draw paired bootstrap interval,
wrong-to-right/right-to-wrong transitions, exact McNemar p, and domain/template
slices. An audit effect replicates only with higher accuracy, a strictly positive
paired lower interval, and McNemar p < .05. Verifier zero regressions remain a
gate property. Any result is constrained to this controlled numerical benchmark
and does not establish a universal scale law or learned scientific knowledge.
