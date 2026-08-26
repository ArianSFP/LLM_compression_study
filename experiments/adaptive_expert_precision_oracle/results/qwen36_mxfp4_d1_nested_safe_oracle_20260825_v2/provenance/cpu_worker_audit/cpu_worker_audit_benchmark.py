#!/usr/bin/env python3
"""Preserve a non-scientific CPU worker scaling audit for Experiment A."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_key_values(path: Path) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (
            line.split() for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def burn(duration: float) -> tuple[int, int]:
    deadline = time.perf_counter() + duration
    count = 0
    value = 1
    while time.perf_counter() < deadline:
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
        count += 1
    return count, value


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--workers", type=int, nargs="+", default=[16, 24, 27, 32])
    args = parser.parse_args()
    if args.duration <= 0 or any(worker <= 0 for worker in args.workers):
        raise ValueError("duration and workers must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty audit directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    script_copy = output / "cpu_worker_audit_benchmark.py"
    shutil.copyfile(Path(__file__).resolve(), script_copy)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    atomic_text(output / "exact_command.txt", command + "\n")

    cgroup = Path("/sys/fs/cgroup")
    cpu_max_raw = (cgroup / "cpu.max").read_text(encoding="utf-8").strip()
    quota, period = cpu_max_raw.split()
    quota_equivalents = None if quota == "max" else int(quota) / int(period)
    lscpu = subprocess.run(
        ["lscpu"], check=True, capture_output=True, text=True,
    ).stdout
    atomic_text(output / "lscpu.txt", lscpu)
    facts = {
        "schema": "pr13_d1_cpu_worker_host_facts_v1",
        "scientific_evidence": False,
        "pilot_nonpromotable": True,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_max_raw": cpu_max_raw,
        "cpu_quota_equivalents": quota_equivalents,
        "cpuset_cpus_effective": (cgroup / "cpuset.cpus.effective")
        .read_text(encoding="utf-8")
        .strip(),
        "sched_affinity_count": len(os.sched_getaffinity(0)),
        "sched_affinity": sorted(os.sched_getaffinity(0)),
        "os_cpu_count": os.cpu_count(),
        "cpu_pressure_before": (cgroup / "cpu.pressure")
        .read_text(encoding="utf-8")
        .splitlines(),
        "cpu_stat_before": read_key_values(cgroup / "cpu.stat"),
        "benchmark_duration_seconds_per_worker": args.duration,
        "worker_grid": args.workers,
        "burn_kernel": "independent scalar 32-bit LCG loops in forked processes",
    }
    atomic_json(output / "host_facts_before.json", facts)

    context = mp.get_context("fork")
    rows: list[dict[str, object]] = []
    header = (
        "workers wall_s aggregate_cpu_s effective_cpu throughput_Miter_s "
        "throttle_events throttle_aggregate_s"
    )
    lines = [header]
    for workers in args.workers:
        before = read_key_values(cgroup / "cpu.stat")
        started = time.perf_counter()
        with context.Pool(workers) as pool:
            results = pool.map(burn, [args.duration] * workers)
        wall = time.perf_counter() - started
        after = read_key_values(cgroup / "cpu.stat")
        aggregate_cpu = (after["usage_usec"] - before["usage_usec"]) / 1e6
        throttle_aggregate = (
            after["throttled_usec"] - before["throttled_usec"]
        ) / 1e6
        row = {
            "workers": workers,
            "wall_seconds": wall,
            "aggregate_cpu_seconds": aggregate_cpu,
            "effective_cpu_equivalents": aggregate_cpu / wall,
            "throughput_million_iterations_per_second": (
                sum(result[0] for result in results) / wall / 1e6
            ),
            "throttled_periods": after["nr_throttled"] - before["nr_throttled"],
            "throttled_aggregate_seconds": throttle_aggregate,
            "cpu_stat_before": before,
            "cpu_stat_after": after,
        }
        rows.append(row)
        lines.append(
            f"{workers} {wall:.6f} {aggregate_cpu:.6f} "
            f"{row['effective_cpu_equivalents']:.6f} "
            f"{row['throughput_million_iterations_per_second']:.6f} "
            f"{row['throttled_periods']} {throttle_aggregate:.6f}"
        )
    atomic_text(output / "raw_output.txt", "\n".join(lines) + "\n")
    atomic_json(
        output / "raw_results.json",
        {
            "schema": "pr13_d1_cpu_worker_scaling_results_v1",
            "scientific_evidence": False,
            "pilot_nonpromotable": True,
            "rows": rows,
        },
    )
    after_facts = {
        "schema": "pr13_d1_cpu_worker_post_facts_v1",
        "scientific_evidence": False,
        "pilot_nonpromotable": True,
        "cpu_pressure_after": (cgroup / "cpu.pressure")
        .read_text(encoding="utf-8")
        .splitlines(),
        "cpu_stat_after": read_key_values(cgroup / "cpu.stat"),
    }
    atomic_json(output / "host_facts_after.json", after_facts)
    readme = """CPU worker audit for PR #13 nested D1 Experiment A

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
"""
    atomic_text(output / "README.txt", readme)

    files = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "schema": "pr13_d1_cpu_worker_audit_sha256_manifest_v1",
        "scientific_evidence": False,
        "pilot_nonpromotable": True,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }
    manifest_path = output / "sha256_manifest.json"
    atomic_json(manifest_path, manifest)
    atomic_text(
        output / "sha256_manifest.json.sha256",
        f"{sha256(manifest_path)}  {manifest_path.name}\n",
    )
    print("\n".join(lines), flush=True)
    print(f"audit_dir {output}", flush=True)
    print(f"manifest_sha256 {sha256(manifest_path)}", flush=True)


if __name__ == "__main__":
    main()
