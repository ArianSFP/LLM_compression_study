CPU worker audit for PR #13 nested D1 Experiment A

Status: NON-PROMOTABLE SYSTEMS PILOT. This directory contains no allocator
candidate, scientific metric, terminal outcome, or promotion evidence. It is
used only to choose the number of single-threaded CPU worker processes.

The host exposes a large CPU affinity mask, but the cgroup CPU quota is the
authoritative capacity limit. The synthetic forked-process kernel checks where
aggregate throughput saturates and records cgroup throttling at each worker
count. Production must still keep OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and
MKL_NUM_THREADS equal to one.

The allocator hashes the CLI worker count into its run identity. Therefore one
worker count must be selected before production and kept fixed across all
layers within a split. This audit does not alter repository code or scientific
inputs.
