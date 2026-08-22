# Refinement-aware rotation audit protocol

## Question and boundary

This PR #13 continuation asks whether an orthogonal change of activation basis
can make the off-device Q2-to-Q4 gate/up correction more top-k compressible.
The resident Q2 base is never rotated or rewritten. For residual
`R = W_Q4 - W_Q2`, basis `Q`, and activation `x`, the streamed path is

```text
R x = (R Q) (Q^T x).
```

The experiment is a train-fit, validation-only basis ceiling. It holds down at
exact Q4 and measures the isolated gate/up correction plus the resulting
nonlinear SwiGLU expert output. It does not evaluate top-8 group allocation,
an H4 activation predictor, causal replay, prefetch latency, routing/logit
agreement, tokens, perplexity, or a downstream task.

The validation cohort is the same four locked hot experts used by the prior
sparse-streaming OMP-gap pilot: experts 62/154/191/108 in layers 0/4/20/39,
with at most 16 deterministic validation occurrences each. Test rows are not
admitted to fit, evaluation, or analysis.

## Basis and encoding controls

The frozen grid is:

- native exact Q2-to-Q4 gate and up rows, selected by independent energy or
  exact nonlinear fixed-state greedy;
- identity paired input coordinates;
- layer-shared uncentered activation PCA;
- layer-shared eigensystems of the uniform and train-router-weighted aggregate
  gate/up residual Grams;
- a layer-shared 32-coordinate block approximate joint-diagonalization basis,
  fit on the 16 highest train-router-mass experts and ranked by the all-expert
  router-weighted Gram;
- a thin per-expert right-singular basis of stacked gate/up residuals;
- a per-expert train-average eigensystem of the first-order Q4 SwiGLU/downstream
  qenergy Jacobian;
- separate per-projection per-expert right-singular bases.

The INT4 shared representations retain 1,024 of 2,048 input directions. A
second, rate-matched control retains all 2,048 directions and requantizes the
transformed atoms to MSE-fitted symmetric four-level INT2 (`-3/-1/+1/+3`).
This separates basis concentration from rank-truncation effects without
wasting a physical codeword on a ternary encoding.
The joint per-expert basis is thin but exact before quantization because a
stacked 1,024-by-2,048 gate/up residual has rank at most 1,024. The separate
projection bases retain 512 directions each.

The deployable-format ceiling uses one symmetric INT4 scale per transformed
gate or up column. Two 512-value INT4 columns contain exactly 512 code bytes,
so a paired coordinate is one physical page; the two FP16 scales are charged
as persistent per-expert metadata. The separate projection control uses one
512-value INT8 column per 512-byte page plus one FP16 scale. The mathematical
FP32 correction eigenbasis is explicitly nonphysical and its x-axis is action
count, not bytes.

The full-rank INT2 layout is page-exact: one transformed coordinate contributes
one 512-value gate column and one 512-value up column, totaling 256 code bytes.
Two adjacent coordinates in the fixed, importance-ordered basis form one
indivisible 512-byte page. Selection ranks these 1,024 fixed pages; it never
assumes that two arbitrary 256-byte half-pages can be dynamically repacked.
Each original column still has its own charged FP16 scale.

Basis storage and transform arithmetic are separate from action traffic:

- dense layer-shared `2048 x 1024` FP16 basis: 4,194,304 bytes/layer and
  2,097,152 MACs once per layer activation;
- full dense layer-shared `2048 x 2048` FP16 basis: 8,388,608 bytes/layer and
  4,194,304 MACs once per layer activation;
- block basis: 64 FP16 `32 x 32` blocks plus 1,024 UINT16 selected-column IDs;
  the full-rank variant stores 2,048 IDs but uses the same blocks;
- dense per-expert basis: 4,194,304 bytes/expert and 2,097,152 MACs/expert;
- separate per-expert bases: the same combined transform arithmetic and
  4,194,304 basis bytes, split across two `2048 x 512` transforms.

The analysis includes a conservative metadata-aware one-bpw control. It starts
from 393,216 bytes/expert, subtracts per-column scales and either the full
per-expert basis or `1/256` of a shared layer basis, then assigns the remainder
to 512-byte actions. This is an equivalent-storage control, not a claim that
resident basis bytes are fetched for every token.

## Locked inputs

- Checkpoint repository: `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`.
- Revision: `7eceff3a9f7e6f916c824d197266d86676bce695`.
- `config.json` SHA-256:
  `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a`.
- `model.safetensors.index.json` SHA-256:
  `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb`.
- Embedded refinement tree SHA-256:
  `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`.

The exact capture was regenerated from the same 12 prompts, 6/3/3
train/validation/test request split, four layers, and maximum length 64 as the
PR #13 input. It is byte-identical to the prior locked capture: SHA-256
`52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931`.
Its environment is recorded in the capture manifest and bound by the fit and
evaluation facts.

## Commands

From `experiments/adaptive_expert_precision_oracle`:

```bash
export PYTHONPATH=src:scripts

python scripts/capture_exact_mxfp4_confirm.py \
  --checkpoint /path/to/qwen36_mxfp4_checkpoint \
  --output /path/to/qwen36_exact_confirm_captures.npz \
  --layers 0 4 20 39 \
  --max-length 64 \
  --revision 7eceff3a9f7e6f916c824d197266d86676bce695 \
  --cpu-only

python scripts/run_refinement_rotation_audit.py \
  --phase fit \
  --config configs/qwen36_mxfp4_refinement_rotation_audit.json \
  --checkpoint /path/to/qwen36_mxfp4_checkpoint \
  --trees results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/dense/selected_trees.json \
  --captures /path/to/qwen36_exact_confirm_captures.npz \
  --output /path/to/refinement_rotation/fit \
  --device cuda:0

python scripts/run_refinement_rotation_audit.py \
  --phase evaluate \
  --config configs/qwen36_mxfp4_refinement_rotation_audit.json \
  --checkpoint /path/to/qwen36_mxfp4_checkpoint \
  --trees results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/dense/selected_trees.json \
  --captures /path/to/qwen36_exact_confirm_captures.npz \
  --fit-dir /path/to/refinement_rotation/fit \
  --output /path/to/refinement_rotation/validation_exact_h4 \
  --device cuda:0

python scripts/analyze_refinement_rotation_audit.py \
  --config configs/qwen36_mxfp4_refinement_rotation_audit.json \
  --fit-dir /path/to/refinement_rotation/fit \
  --evaluation-dir /path/to/refinement_rotation/validation_exact_h4 \
  --output /path/to/refinement_rotation/analysis
```

The fit/evaluation output directories must not already exist. Fit facts bind
the config, capture, tree, checkpoint config/index, device, and every layer
artifact hash. Evaluation rejects a different fit provenance. Analysis rejects
incomplete or duplicate grids, non-validation rows, nonfinite required values,
physical actions that are not exactly one 512-byte page, and a mathematical
oracle mislabeled as physical.

## Verification

Run the focused suite:

```bash
PYTHONPATH=src:scripts pytest -q tests/test_refinement_rotation.py
```

The tests cover exact correction-basis reconstruction, diagonal/OMP identity,
uncentered PCA, transformed endpoint identity, INT2/INT4 byte accounting,
FP16 scale-underflow handling, indivisible two-coordinate INT2 pages, joint
diagonalization, block orthogonality, nonlinear native-state closure, the
train-only output-Jacobian basis, and analyzer summaries.
