# Neuron-selector distillation protocol

This study is stacked on PR #7 at commit
`9ef21d519a4f2cd844e9fa75d04cd601e37d5a11`. It does not alter the embedded
Q2→Q3→Q4 codec, selected trees, checkpoint revision, or request split. It
investigates two questions only: whether gate/up and down refinement should be
allocated independently per SwiGLU unit, and whether the exact complete-unit
score can be predicted from a compact gate/up residual-response model.

Recovery always means exact sequential complete-expert qenergy recovery. It
does not mean task accuracy, logit fidelity, router fidelity, or token quality.

## A. Factorized neuron action space

For each unit, the four exact states are

```text
y00 = d2 * h2       y10 = d2 * h4
y01 = d4 * h2       y11 = d4 * h4
```

where `hℓ = SiLU(Gℓ x) * (Uℓ x)`. Gate/up refinement costs two 512-byte
pages; down refinement costs one. Exact marginal-per-page fixed-greedy may
take `00→10→11` or `00→01→11`. Coefficients are fixed, no least-squares refit
is performed, and the implementation maintains exact correlations under the
nonidentity future-proxy qmetric. Each headline physical budget is run as an
independently constrained path at 0.5, 0.75, and 1.0 bpw.

The controlled baselines are coherent three-page unit packets from PR #7 and
the frozen exact nonlinear 64×16 tile validation frontier. Factorized neurons
promote only if their validation median and p10 at one bpw are within 1.0 and
1.5 recovery points, respectively, of the frozen tile oracle.

## B. Compact sufficient statistic and response predictor

For a coherent unit correction `c = d4*h4 - d2*h2`, store

```text
A = d4' G d4      B = d4' G d2      C = d2' G d2
s = A*h4^2 - 2*B*h4*h2 + C*h2^2
```

The three vectors are normalized, encoded, and paired with one FP32 scale
per vector, avoiding FP16 overflow. The runtime unknown is reduced to the 512
Q4 hidden scalars.

The fitted response model is

```text
zg = Bg x             delta_g_hat = Ag,e zg
zu = Bu x             delta_u_hat = Au,e zu
h4_hat = SiLU(g2 + delta_g_hat) * (u2 + delta_u_hat)
```

Layer-shared joint and separate gate/up bases are compared at ranks
8/16/32/64/128. A sampled per-expert joint basis is a diagnostic upper bound
and can never be promoted as bank-wide deployment evidence. Fits use training
requests only. The same layer training activations are deliberately applied
to every sampled expert as response-label augmentation; validation and test
activations, masks, scores, and recoveries are excluded. Both exact-checkpoint
training-only and combined exact-plus-cross training-only fits are evaluated.

Layer transforms are FP16. Expert synthesis factors are evaluated in FP16,
FP8 E4M3FN, and per-row INT8. Metadata bpw includes amortized layer transforms,
expert synthesis factors, and the A/B/C statistic. Runtime is a late H0
second-pass selector because it consumes the true resident Q2 responses; no H4
claim is made.

## Candidate prefetch experiment

The predicted ranking fetches 192/224/240/256 coherent unit packets. Exact
A/B/C scores rerank the fetched set at H0 and apply 192 units. Physical bytes
charge every fetched three-page packet, while logical bytes charge only the
applied packets. Promotion requires the frozen median/p10 utility, recovery,
overfetch, and metadata gates in the JSON config.

## Split discipline and bounded expansion

1. Fit both frozen training cohorts. The fit facts must state that no
   validation or test rows were used.
2. Run every available exact-checkpoint validation occurrence for the locked
   experts, up to 16 per expert.
3. Generate a canonical validation-only promotion artifact. The held-out
   runner independently recomputes all gates and rejects malformed hashes,
   coverage, booleans, or a nonwinning configuration.
4. If a layer-shared predictor passes, an activation-top-k continuation may be
   run on validation only. If shared bases fail but the sampled per-expert
   upper bound passes, 4/8 cluster bases may be investigated on validation
   only. Neither continuation may mutate the frozen base config.
5. Evaluate held-out exact-checkpoint test rows only after a decision is
   frozen. Cross-reference rows are sensitivity evidence and are never pooled
   with the primary cohort.

The exact factorized oracle may promote independently of the response
predictor. A negative predictor result is a valid conclusion and stops its
held-out expansion.

## Physical accounting

The expert has 3,145,728 weights. One correction bpw is 393,216 bytes or 768
512-byte pages. A coherent unit packet is 1,536 bytes. Selector metadata is
reported separately from fetched suffix traffic, and the storage multiplier
is `(4.25 + 2.0 + selector_metadata_bpw) / 4.25`. `selector_bytes_read` is a
unique-tensor-footprint lower bound, not measured hardware traffic.

## Frozen implementation checkpoint

- config SHA-256: `7c19d1994d91492e921aad4e214c9f3421db4c970b1f6eddacef49d503a90fd1`
- selector core: `26ce5de7f6b3f28eb77b4119964c9f911fecfe3232d7e5171c17040671283a11`
- runner: `e796d5c2bb0a03676bc3b6631811a7b2810755edb0da789684e35cd7cc7ae2f5`
- analyzer wrapper: `af3bc8158b4da891f16b56051e3ccac2672c635876d26f5a6980215c8f6aa82c`
- promotion analyzer: `d346cc367107969d36dca256ebc606a088a9ebc5e0542e80466c7a799febfdb7`
- focused tests: 22 passed
- integrated tests: 143 passed with one pre-existing numexpr warning

Later corrections and their replacement hashes must be recorded in the
execution ledger before they are used for scientific evidence.
