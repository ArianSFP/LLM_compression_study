# Decode and Resident-Metadata Accounting Memo

Date: 2026-08-19  
Scope: independent analytical accounting for the baseline and Designs A--E  
Status: estimates from the implemented binary formats; no decode kernel or GB10 latency was measured

## Accounting basis

One routed expert contains 3,145,728 weights: 1,048,576 each for gate, up, and down. The complete routed model contains 40 x 256 = 10,240 experts and 32,212,254,720 expert weights. The common resident parent is 884,736 bytes/expert (2-bit parents plus one E8M0 byte per 32 weights), or exactly 2.25 bpw before hierarchy metadata.

Each exact progressive book is the implemented 32-byte binary object: eight bytes of packed leaf-to-`(parent,r1,r2)` digits followed by twelve FP16 Q2/Q3 centroids. Per-expert selector/modifier files use the implemented 64-byte `CGBMXF4` header and 16-byte projection alignment. The payload sizes below happen to be multiples of 16, so B--E have zero alignment padding.

The representative accounting variants are:

- baseline: one 32-byte book per projection, globally shared;
- A: 512 independent books per projection/expert, one per 2,048-weight functional vector;
- B: 16 families per native 32-weight block, tables shared per layer/projection;
- C: eight families per 32-weight block plus four FP16 values (eight bytes) per functional vector, tables shared per layer/projection;
- D: four families per 16-weight subgroup, tables shared per layer/projection;
- E: 64 trained families per 64-weight group with an eight-bit selector. The byte can address 256 families; 64 is the largest currently implemented fit arm.

Research-state NPZ files and JSON manifests are intentionally excluded from resident deployment bytes. Every charged binary table, selector, modifier, header, and alignment byte is included.

## Serialized resident cost

| Design | Selector/modifier or local-table bytes/expert | Header bytes/expert | Shared tables, full model | Amortized metadata bytes/expert | Added metadata bpw | Total resident bpw |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0 | 0 | 96 | 0.009375 | 0.000000024 | 2.250000024 |
| A | 49,152 | 0 | 0 | 49,152 | 0.125000000 | 2.375000000 |
| B16/G32 layer | 49,152 | 64 | 61,440 | 49,222 | 0.125178019 | 2.375178019 |
| C8/G32 layer | 49,152 | 64 | 30,720 | 49,219 | 0.125170390 | 2.375170390 |
| D4/G16 layer | 49,152 | 64 | 15,360 | 49,217.5 | 0.125166575 | 2.375166575 |
| E64/G64 layer | 49,152 | 64 | 245,760 | 49,240 | 0.125223796 | 2.375223796 |

For B, changing sharing scope changes only table bytes: global/projection is 1,536 bytes model-wide, three expert-frequency clusters/projection are 4,608 bytes, layer/projection is 61,440 bytes, and expert/projection is 15,728,640 bytes. The per-expert selector files remain 49,216 bytes. For E, the full-model layer/projection table footprints are 61,440/122,880/245,760 bytes for 16/32/64 trained families. A hypothetical dense 256-family arm would use 983,040 table bytes while retaining the same selector stream.

The 64-byte header adds 655,360 bytes over all 10,240 experts. It is why the serialized B--E rates are slightly above 2.375 bpw even before shared tables.

## Decode operations

All designs can decode Q2 or Q3 with one reconstruction-table lookup per weight after selecting a table base. Exact Q4 may also remain one lookup/weight if the loader expands each book's inverse `(parent,r1,r2)->leaf value` table once. A literal two-step inverse-map plus E2M1 lookup would add one lookup/weight but requires no additional serialized metadata.

The reported table-switch frequency is the maximum table-base selection cadence, not the measured fraction of adjacent groups whose family IDs differ:

| Design | Table-base selections/weight | Selector handling | Additional decode work |
|---|---:|---|---|
| Baseline | 1 / 1,048,576 | none | one projection-global base |
| A | 1 / 2,048 | none | advance to the functional-vector table |
| B | 1 / 32 | four-bit packed ID | nibble extraction plus family-stride address |
| C | 1 / 32 | three-bit packed ID | bitstream extraction; apply vector affine modifier |
| D | 1 / 16 | two-bit packed ID | twice B's table-base cadence |
| E | 1 / 64 | direct eight-bit ID | byte load; no packed-bit extraction |

C adds an estimated multiply and add per Q2/Q3 weight (two scalar FLOPs, potentially one FMA instruction) and reads one eight-byte modifier per 2,048-weight vector. The integer/address-operation ranges in the CSV are implementation estimates, not instruction-counter measurements. They include group index/selector extraction/table-stride formation but exclude the common parent/refinement unpack and E8M0 scale indexing.

## GEMV fusion assessment

- Baseline is straightforward: a projection-constant reconstruction table can be hoisted outside the inner loop.
- A is fusion-friendly for gate/up row-major traversal because a table remains constant for one output row. Down uses a new table per input column, which aligns with column-conditioned accumulation but requires a different traversal or table-pointer stream. Its 48 KiB/expert table footprint is much larger than a family LUT but changes only every 2,048 weights.
- B is the cleanest local-family option. Its G32 choice is aligned with the native E8M0 block, so the selector, scale, and 32 leaf/parent codes can be loaded as one decode unit. The three 16-family projection tables occupy only 1.5 KiB for a cold layer scope.
- C retains G32 alignment, but three-bit IDs can straddle bytes and its per-vector affine correction adds parameter loads and arithmetic. Fusion is feasible but meaningfully more complicated than B.
- D is feasible but chooses two codebooks inside each native G32 scale block. It doubles selector/table-base cadence and introduces half-block control boundaries, making it the least attractive of the family designs unless quality gains are material.
- E has the cheapest selector decode (one byte per 64 weights) and half B's base-selection cadence. A G64 family spans two native G32 scale blocks, so scale handling still occurs twice. The 64-family layer working set is 6 KiB across projections and remains small; a dense 256-family set would be 24 KiB per layer across projections.

## GB10 arithmetic-intensity implication

For comparability with the existing report, useful arithmetic uses the same two-FLOP convention: 6,291,456 FLOPs for the resident expert GEMV and 12,582,912 FLOPs at one physical streamed bpw. The estimate assumes all resident parent, scale, and metadata bytes plus 393,216 streamed bytes are read once per expert invocation. It does not model cache reuse of shared tables.

| Design | Resident-only useful FLOP/byte | Useful FLOP/byte at one streamed bpw | Change at one bpw vs baseline |
|---|---:|---:|---:|
| Baseline | 7.1111 | 9.8462 | reference |
| A | 6.7368 | 9.4815 | -3.704% |
| B | 6.7363 | 9.4810 | -3.709% |
| C | 6.7364 | 9.4810 | -3.709% |
| D | 6.7364 | 9.4810 | -3.708% |
| E | 6.7362 | 9.4809 | -3.710% |

The 0.125-bpw resident investment therefore reduces useful arithmetic intensity by about 3.7% at one streamed bpw if all added metadata is charged as a read. C's modifier arithmetic is deliberately excluded from "useful" FLOPs: counting extra decode work would raise numerical FLOP/byte without improving GEMV work and could hide an instruction-throughput cost.

These remain analytically memory-bound points under the prior GB10 roofline assumptions, but that is only a classification. Cache behavior, selector unpack, indirect addressing, instruction issue, register pressure, output writes, and fused-kernel scheduling are unmeasured. This memo makes no GB10 latency or throughput claim.

Machine-readable values are in `decode_accounting.csv` beside this memo.
