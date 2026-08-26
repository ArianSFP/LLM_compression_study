# Non-promotable calibration attempt: raw unanchored route gate

This directory preserves the first production calibration attempt from commit
`843dec86b33cbffa1969e12108ab7fb9baf9c417`. It is non-promotable and stopped
during calibration layer 6, request 23/32, before any layer-6 scientific files
or global run facts were finalized.

The hard failure was:

`RuntimeError: stored-hidden cached decode changes ordered target IDs`

The triggering request was `9118185608909195810`, layer 6, position 33. The
unanchored slice and captured route contained the same eight experts but
swapped tied experts 123 and 216 within ranks 7/8. The maximum constant anchor
was 0.03125; after anchoring, logits were bit-identical and CUDA top-k matched
the authenticated ordered target exactly.

A subsequent read-only audit covered all 768 request/layer contexts. It found
six unanchored ordered-route mismatches and one unanchored membership mismatch,
all at layer 6. All 768 anchors remained within 0.0625, all 768 anchored logits
were bit-identical, and all 768 anchored ordered routes matched. This showed
that the unanchored route was an over-strict implementation gate rather than a
declared scientific target. Commit `74f0ce2ce04555c41beae701389772b33f0df483`
corrects the runner to serialize raw route drift diagnostically while retaining
anchored bit equality and anchored ordered top-8 as hard gates.

Completed layer 0, 1 and 4 artifacts are retained here only as failure evidence.
They must not be mixed with the corrected production run because their sealed
Python code identity differs.
