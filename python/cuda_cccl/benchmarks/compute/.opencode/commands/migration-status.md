---
description: Summarize CUB vs Python benchmark migration status
---

Generate a concise migration status summary for cuda.compute benchmarks.

Requirements:
- Source of truth for coverage is CUB benchmarks under `cub/benchmarks/bench/`.
- Status target is Python benchmarks under `python/cuda_cccl/benchmarks/compute/`.
- Include both migrated and unmigrated CUB benchmarks.
- For each migrated benchmark, provide a one-line status note (API parity + key delta).
- For each unmigrated benchmark, provide a one-line reason it is not yet possible.
- IMPORTANT: inspect Python source APIs (e.g., `python/cuda_cccl/cuda/compute/` and related packages) to confirm whether the required primitives exist before claiming a migration gap.
- Cross-check with `analysis/migration_summary.md`; if it appears stale, say so explicitly.

Output format:
1) Migrated (group by top-level category): list `bench_name — status line`.
2) Unmigrated (group by top-level category): list `bench_name — reason`.
