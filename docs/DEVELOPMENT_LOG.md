# Development log

I used only the 18 development cases while changing the interface. I did not
inspect the 72 test outcomes until the final pipeline was frozen.

## Attempt 1: free generation with a final marker

I allowed 128 new tokens and asked Qwen to end with `FINAL: <letter>`. It spent
the budget writing formulas and all nine sampled answers were truncated before
the marker. That measured verbosity and truncation, not scientific accuracy.

## Attempt 2: answer-only generation

I asked for only the final marker and reduced the budget to 24 tokens. The model
still began long explanations; all 18 development answers missed the marker.
Prompt insistence did not make free-text parsing reliable.

## Attempt 3: forced-choice logits without an executed audit

I appended `FINAL:` as an assistant prefill and scored the next-token logits of
A/B/C/D. This fixed parsing and produced 61.1% on the development set, but the
"audit" instruction alone left every choice unchanged. The model never actually
generated or consumed a second calculation, so this was not a real loop.

## Frozen design

The final pipeline makes the intermediate computation explicit:

1. generate a bounded scientific scratchpad;
2. score the direct label with forced-choice logits;
3. generate a second audit using either no external feedback or a deterministic
   error category;
4. score the revised label with the same forced-choice method.

On all 18 development cases, direct accuracy was 72.2%, intrinsic audit was
77.8%, and verifier-gated accuracy was 83.3%. I then froze the code and ran the
72 held-out cases once.

The raw development JSON files are not part of the claimed result because two
belong to invalid measurement designs. The test artifact contains the complete
claimed run.
