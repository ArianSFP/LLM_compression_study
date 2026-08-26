# Qwen36 MXFP4 nested D1-safe Experiment A

This compact package contains the held-out nested D1-safe oracle result.
Experiment A **failed** its promotion gate; Experiment B was not started.

Start with the
[canonical report](analysis/D1_NESTED_SAFE_ORACLE_REPORT.md) and the
[full methodology](../../D1_NESTED_SAFE_ORACLE_METHODS_20260825.md).

The package includes:

- complete finalized global outcome tables;
- all 17 byte-reproduced primary analysis outputs;
- reproducible post-hoc failure diagnostics;
- evaluation allocation metrics, parity and layer facts;
- calibration parity, layer facts, frozen parameters and calibration evidence;
- a losslessly compressed complete allocation manifest;
- production logs, run facts and CPU worker audit.

The decompressed allocation manifest must have SHA-256
`1bdd2d1c047416ff4649d65306733af5a760ed7aa4295d293cd06c034679ca59`.
Use:

```bash
zstd -d d1_nested_evaluation_allocation_manifest.json.zst
sha256sum d1_nested_evaluation_allocation_manifest.json
```

Multi-gigabyte candidate banks, full timing traces, exact captures and the raw
uncompressed manifest remain on network storage at:

```text
/workspace/pr13_d1_nested_safe_oracle_20260825_v2
```

`artifact_hashes.sha256` authenticates every packaged file except itself and
`PACKAGE_MANIFEST.json`; the latter records the package inventory and source
location.
