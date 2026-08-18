# Primary multiple-description result

This directory contains the paired A-first, bidirectional A/B and A/B/Q parity
oracle on 70 held-out expert invocations from six fresh exact-checkpoint test
requests. The primary physical page size is 512 bytes.

`metrics.parquet` contains one complete-expert result per policy, invocation and
budget. Exact H0 labels are in `selected_description_actions/`, partitioned by
layer only to remain below GitHub's per-object size limit. Concatenating the
four files reproduces the original monolithic 4,410-row table without loss.

The raw capture is not committed. Its schema and SHA-256 are in
`capture_manifest.json`, and the preserved network path is recorded in the
execution ledger.
