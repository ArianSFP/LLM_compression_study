# Parity shared-layout control

This controlled run exposes A, B and `Q=A xor B` descriptions but uses the
identical training-only A/B co-selection layout as the primary bidirectional
policy. It therefore isolates description utility from layout perturbation.

The parity allocator selected Q zero times. All 490 invocation-budget results
are bit-for-bit identical to bidirectional A/B in recovery and physical bytes.
The complete exact action table is retained in
`selected_description_actions.parquet`.
