---
name: cccl-benchmarking
description: Use when asked to build/run a CCCL benchmark (CUB, Thrust, or libcu++), measure performance, or compare benchmark results between branches/commits.
---

Read `docs/cpp/cub/benchmarking.rst` before building, running, or comparing benchmarks. Despite its location under
`docs/cpp/cub/`, it applies to Thrust and libcu++ benchmarks too.

Always run benchmark binaries with `-d 0 --stopping-criterion entropy --md <filename>.md --json <filename>.json`, and
keep both the markdown and JSON files for later processing (e.g. with `nvbench_compare.py`).
