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
independently constrained path at 0.5, 0.75, and 1.0 bpw. A path may traverse
a negative prerequisite to unlock a profitable complementary transition, but
the reported state is the best cumulative prefix under the page cap rather
than blindly applying the last evaluated transition.

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
row-scaled FP8 E4M3FN, and per-row INT8. Metadata bpw includes amortized layer transforms,
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
held-out expansion. In the executed study both families stopped, so no test or
cross-reference evaluation was launched.

## Physical accounting

The expert has 3,145,728 weights. One correction bpw is 393,216 bytes or 768
512-byte pages. A coherent unit packet is 1,536 bytes. Selector metadata is
reported separately from fetched suffix traffic, and the storage multiplier
is `(4.25 + 2.0 + selector_metadata_bpw) / 4.25`. `selector_bytes_read` is a
unique-tensor-footprint lower bound, not measured hardware traffic.

## Frozen result

Corrected validation covered 69 routed invocations over all twelve sampled
experts. The frozen matched comparison uses the same 55 invocations available
to the PR #7 tile control.

- Factorized G/D fixed-greedy at the one-bpw cap: p10/median
  `0.869842/0.932810`; exact 64×16 tile: `0.935183/0.966025`. The family
  stopped. At 0.5 bpw factorization improves coherent packets, but the gain
  disappears by 0.75 bpw and reverses at 1.0 bpw.
- Best bank-deployable response candidate under 0.35 metadata bpw: combined-
  train, separate gate/up, rank 64, row-FP8. With 256 fetched and 192 applied
  units it reaches p10/median recovery `0.755966/0.888333` and independent-
  score utility retention `0.916435/0.956179` at `0.184926` metadata bpw.
  It stopped on recovery; even the sampled per-expert upper bound failed.
- The exact A/B/C statistic itself needs 3,084 bytes/expert (`0.007843` bpw).
  The negative result is specifically the proposed h4 response predictor plus
  independent-unit scoring, not the algebraic identity.
- Frozen promotion SHA-256:
  `7430a885d6dfa303ac1a54357fa7a86f54bd9000df46d23f2727fd45fe961bc4`.
- No activation-top-k/cluster continuation, test run, cross sensitivity, or H4
  training was performed after the stop.

## Initial pre-run implementation checkpoint (historical)

- config SHA-256: `41eb0e21cdb016017d8f3b4a48bcdd4425323c463d96afedd0bf585cfbedf3d4`
- selector core: `24c296a0fe3d5390543bedb7847f8d60bcb3b7dfd3ebfce30eaccba7b294a871`
- runner: `0a8cbf20d2e46777af2853087bfcacf48eae9ade3298b5168b27d3a94d3b78de`
- analyzer wrapper: `af3bc8158b4da891f16b56051e3ccac2672c635876d26f5a6980215c8f6aa82c`
- promotion analyzer: `d346cc367107969d36dca256ebc606a088a9ebc5e0542e80466c7a799febfdb7`
- focused tests: 26 passed
- integrated tests: 147 passed with one pre-existing numexpr warning

Later corrections and their replacement hashes must be recorded in the
execution ledger before they are used for scientific evidence.

## Executed implementation checkpoint

- config: `41eb0e21cdb016017d8f3b4a48bcdd4425323c463d96afedd0bf585cfbedf3d4`
- selector core: `29b691289ac82694144c412caeef4e44ee98f6bd6e1e9ef66731c6a269123f94`
- evaluation runner: `9d3a44cf2e4fa82ee533c482ac52073a9bad7968aa411fdc34dace8450d59e29`
- final analyzer: `d8d257230a6fc1fe46526b38de790ac5d211b6f1a7d6cd6180edfda6a69a1f20`
- fit bundle manifest: `73b0285be3e3611fa44ac0995bd69c3b4390cd02b0fa9ac03a9626397c878d60`
- focused tests: 32 passed;
- integrated tests: 153 passed in 21.67 seconds, with the inherited
  pandas/numexpr warning only;
- final analysis rerun: byte-identical CSV, JSON, Markdown, PNG, SVG, and
  manifest outputs after fixing Matplotlib's SVG hash salt and date metadata and normalizing SVG trailing whitespace.
