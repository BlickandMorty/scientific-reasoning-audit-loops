# SmolLM2 execution boundary

The SmolLM2 balanced-label replication protocol was publicly committed and CI
verified before attempting local execution. On 2026-08-28, the public ungated
model metadata and small configuration files downloaded successfully, but the
multi-gigabyte weight transfer stalled before a weight file was available or
CUDA inference began. A normal retry with Hugging Face Xet disabled also stalled
before model load.

I stopped only the two stalled local download processes. No result artifact was
created, no model output was inspected, and the protocol was not altered or
silently redirected to another model. This is an environment/download boundary,
not a failed scientific replication. A future run may resume the same frozen
SmolLM2 protocol only after the public weights are completely available in the
local cache.
