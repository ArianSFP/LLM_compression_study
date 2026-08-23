# PR #13 causal downstream replay execution ledger

Date: 2026-08-23

Status: reconstruction and isolated impulse phases complete; end-to-end
streamed quality phase deferred

## Scope and locked lineage

This study starts from frozen PR #13 all-layer commit
`089deb4bd41eb71864779ced0cbb3db41e9dcb63` (original PR #13 commit
`dfba3748e51d6916dc1f28cb1d3b3250188adbbe`). Its run ID is
`qwen36_mxfp4_causal_downstream_replay_20260823_v1` on branch
`agent/mxfp4-causal-downstream-replay`.

The completed scope reconstructs every stored PR #13 selected precision state,
audits local qenergy and page parity, and injects each reconstructed error at
one layer before executing the complete downstream sequence tail. It does not
run all layers approximately in one sequential forward and therefore does not
claim a bpw-to-perplexity curve, cross-layer interaction Gamma, or
repair-one-layer importance.

Locked inputs:

- checkpoint revision:
  `7eceff3a9f7e6f916c824d197266d86676bce695`;
- checkpoint config/index SHA-256:
  `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a` /
  `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb`;
- capture manifest/identity-grid SHA-256:
  `40c06b1495e52f689f4d95217ccec70711a69166d63067c3dcdcb4589078cc4d` /
  `075c851bbda32ad3009d4bfe56815a03458524b304ecb4165bb4c8f119538f00`;
- selected-tree SHA-256:
  `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`;
- expert-allocation/group-frontier SHA-256:
  `19c2dc0f94cd9a8237dd866b1a1997c4edcdf6a66457f5b924c4a1e9604fb0c4` /
  `f74e1225789c6502ed7818caa2cad2c50e74633254297b0fb6ebb5c5886f4db6`;
- average-rate facts/factor-manifest SHA-256:
  `e0f58de6c5345885488068d5efb24c063f7a910d4b7fc32fb050f9d0f5e79d19` /
  `60204faca5d9d7974e035ff05088938af942fbd5e0d313c030dddc6748157069`.

Only validation requests `mxfp4-confirm-006`, `-007`, and `-008` are used.
They provide 32 complete captured token positions per layer and 1,280
token/layer groups over 40 layers. Test rows were not admitted or used.

## Pod audit

The supplied pod exposed an NVIDIA RTX PRO 6000 Blackwell Workstation Edition
with 97,887 MiB VRAM, compute capability 12.0, and a 600 W limit; two AMD EPYC
9554 sockets with 128 physical / 256 logical CPUs; 1 TiB RAM; and a 500 GB
local root filesystem plus the larger `/workspace` mount.

Execution used Python 3.12.3, PyTorch 2.8.0+cu128, NumPy 2.1.2, pandas 3.0.5,
Transformers 5.15.0, compressed-tensors 0.18.0, and CUDA runtime 12.8. The exact
checkpoint was copied to `/root/pr13_causal_checkpoint` and both checkpoint
hashes were rechecked.

The dequantized BF16 model used 65.464 GiB allocated, 81.434 GiB reserved, and
65.779 GiB peak allocated memory. Model load took 51.14 seconds. This confirms
that the PRO 6000 was appropriate; a 24 GiB RTX 3090 cannot hold this exact
reference model on one GPU.

## Reconstruction audit

An initial network-checkpoint attempt with five concurrent 24-thread jobs was
storage-I/O limited. It was stopped cleanly, the checkpoint was copied locally,
and the run resumed with 16 concurrent layers and four threads per layer. One
incomplete pre-resume temporary layer-2 file was identified exactly and
removed; no completed artifact was deleted or overwritten.

Final facts:

- validation rows: 1,280;
- group/rate reconstructions: 5,120;
- routed expert comparisons: 40,960;
- selected page vectors exact: true;
- max expert/group qenergy absolute error:
  `1.7763568394002505e-15` / `5.551115123125783e-17`;
- max all-Q4 reconstruction absolute error:
  `4.440892098500626e-15`; and
- causal replay admitted: true.

The reconstruction manifest SHA-256 is
`7a51457c661a9b9b7f9067345e89e26f127f80faa717d746dd8a5857b50476c3`.

## Historical trajectory and paired replay

A fresh full GPU forward did not reproduce the old CPU capture bit-for-bit:
captured-`xplus` RMSE was `0.0054813523`, max hidden error `0.40625`, max router
error `0.421875`, exact ordered top-8 fraction `0.4625`, mean top-8 set overlap
`0.97441406`, and top-1 route agreement `0.975`.

The same-process zero-hook rerun was exactly equal for hidden states, routers,
and logits. A separate current-host CPU tail also failed historical identity
(RMSE `0.0044393`, max error `0.4375`, ordered-route fraction `0.56923`). Thus
the strict historical identity gate remains failed; it was not weakened.

The admitted design is paired tail replay:

1. assemble the complete injection-layer sequence from base `xplus` rows plus
   the last row's exact H1--H4 sequence positions;
2. run no-op and perturbed tails from that same captured BF16 state in one
   resident GPU process; and
3. preserve paired-no-op versus historical-capture drift as separate columns.

Every paired tail was repeated with identical input. Maximum repeated-tail
hidden, router, and logit differences were exactly zero for all 40 layers.

## Impulse execution

The initial operating points are 384 and 749 mean pages per expert, or
0.5234781901 and 0.9987386068 charged bpw. The three complete sequences receive
the stored local PR #13 error at their captured positions. The tail includes
full attention, later MoE layers, final norm, and the LM head.

The first layer-0 sentinel exposed an accounting-label issue: an all-token
numerator used an injected-position denominator and produced 0.72 instead of
1.0 at injection. Its three files were moved to the recoverable `sentinel_v0`
archive. The corrected runner stores explicit all-token and injected-position
energies; both corrected layer-0 coefficients are exactly 1.0.

The remaining 39 layers ran in one model residency. Final facts:

- completed layers/failures: 40/0;
- propagation/final-quality rows: 4,920/240;
- candidate-tail time summed over sidecars: 56.72 seconds;
- repeated baseline-tail time: 57.34 seconds; and
- layer-sidecar wall time: 188.68 seconds.

Combined propagation/quality SHA-256:

- `429abe5a5cda87c41825dc410db96bee5b76a6cd3a2e8a421a3114183585bc40`;
- `343b324908506b725092e1ff478ac6e43c0f4ba86841bdf5e36185ae055f943a`.

## Analysis outcome

Analysis uses a deterministic 10,000-resample bootstrap over independent
validation sequences, not tokens.

| Quantity across 40 isolated layers | 0.523478 bpw | 0.998739 bpw |
| --- | ---: | ---: |
| Median initial realized error energy | `2.431e-3` | `3.304e-4` |
| Median final hidden MSE | `1.069e-4` | `9.423e-5` |
| Median final top-8 route churn | `1.420%` | `1.420%` |
| Median final-logit KL | `1.515e-3` | `1.403e-3` |
| Median absolute delta NLL | `0.00594` | `0.00501` |

The high-rate local error is much smaller, but final absolute disturbance is
not proportionally smaller. Small BF16 perturbations can cross rounding and
routing boundaries, so relative propagation must be read with absolute MSE,
route churn, and logit KL. Signed NLL changes in both directions on this
three-sequence cohort are not a population-quality estimate.

The first full analysis wrote tables and plots, then stopped before the report
and manifest because the report used provisional field `group_reconstructions`
instead of finalized `reconstructions`. Inputs were unchanged. The field was
corrected, byte compilation passed, and the full 10,000-bootstrap analysis
reran from hashed inputs. The final manifest hashes ten outputs.

Verification:

- causal tests: 5 passed locally and remotely;
- complete experiment suite: 349 passed, 1 skipped, with only the pre-existing
  pandas/numexpr version warning;
- byte compilation: passed locally and remotely;
- all 40 reconstruction shards and 40 impulse triplets hash-checked;
- repeated-tail equality: exact across all layers;
- combined numeric fields: finite; and
- `git diff --check`: passed.

The compact repository bundle contains combined Parquet tables, run facts,
aggregate CSVs, report, and plots. Reconstruction shards, per-layer impulse
files, the 27 MiB GPU baseline, checkpoint, and raw capture remain externally:

`/workspace/pr13_causal_downstream_replay_20260823_v1`

## Network-storage handoff

At the owner's request, every staged implementation, configuration, test,
compact result, report, plot, manifest, and ledger file was packaged under the
study's persistent network archive in both compressed and directly readable
forms:

`/workspace/pr13_causal_downstream_replay_20260823_v1/repository_handoff`

The upload is verified at two levels: the local and remote compressed-archive
SHA-256 values match, and deterministic digests over the 25 individual local
and unpacked remote file hashes match. Final checksum sidecars live beside the
network archive. The original 138 MiB capture, 23 GiB PR #13 archive, and 247
MiB full causal study archive remain on the same network mount.


## Shutdown

After compact artifacts and hashes were verified locally, RunPod
`93yrh6ok69il2a` (`hurt_lavender_wolverine`) was first stopped with
`runpodctl pod stop` at 2026-08-23 00:46:53 UTC. At the owner's request it
was resumed at 00:50:28 UTC solely to copy and verify the repository
handoff, then stopped again at 00:54:12 UTC. A final archive/ledger consistency
refresh resumed it at 00:55:46 UTC. After checksum replacement it was stopped
again, and the final all-pod listing reported `EXITED`. The raw capture and
both study archives reside on the persistent `mfs#eu-cz-1.runpod.net`
`/workspace` mount. The disposable local-root checkpoint can be reconstructed
from the locked external archive.


## Next gate

The next experiment needs a supplementary full-sequence held-out capture and a
genuine sequential streamed model at all four PR #13 rates. Frozen-route and
live-route oracle modes must remain separate. Predicted-H4 is out of scope;
later prediction work begins at H1.
