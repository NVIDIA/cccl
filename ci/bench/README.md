# CUB Benchmark Compare Scripts

This directory contains the scripts used by `.github/workflows/bench_cub.yml` to compare CUB benchmark results between two code states.

## Scripts

- `ci/bench/cub.sh`: CI-oriented wrapper that calls `ci/bench/compare_git_refs.sh`.
- `ci/bench/compare_git_refs.sh`: checks out `<base-ref>` and `<test-ref>` in temporary worktrees, then forwards all remaining args to `ci/bench/compare_paths.sh`.
- `ci/bench/compare_paths.sh`: configures/builds/runs common `cub.bench.*` targets in two source trees and runs `nvbench_compare.py` on produced JSON outputs.
- `ci/bench/parse_bench_matrix.sh`: parses `ci/bench.yaml` and emits a dispatch matrix JSON object for `.github/workflows/bench_cub.yml`.

## Usage

Compare two refs:

```bash
"./ci/bench/cub.sh" "origin/main" "HEAD" "^cub\\.bench\\.copy\\.memcpy\\.base$"
```

Forward additional options (parsed by `compare_paths.sh`):

```bash
"./ci/bench/cub.sh" \
  "origin/main" \
  "HEAD" \
  --arch "native" \
  --nvbench-args "..." \
  --nvbench-compare-args "..." \
  "^cub\\.bench\\.reduce\\..*$"
```

Compare already checked-out trees:

```bash
"./ci/bench/compare_paths.sh" \
  "/path/to/base/cccl" \
  "/path/to/test/cccl" \
  --arch "native" \
  "^cub\\.bench\\.copy\\.memcpy\\.base$"
```

## Workflow Inputs

In `.github/workflows/bench_cub.yml`:

- If `raw_args` is non-empty, it is parsed and passed directly to `ci/bench/cub.sh`.
- Otherwise, args are assembled from `base_ref`, `test_ref`, `arch`, `filters`, `nvbench_args`, and `nvbench_compare_args`.
- Malformed quoted input (for example unmatched quotes) fails the workflow step.

## Artifacts

`compare_paths.sh` writes a run directory under `${CCCL_BENCH_ARTIFACT_ROOT:-$(pwd)/bench-artifacts}` containing:

- per-target JSON and markdown outputs for base/test runs,
- grouped build logs (`build.base.log`, `build.test.log`), per-target run logs, and per-target compare logs (`compare.<target>.log`),
- `summary.md` with run metadata and per-target collapsible full `nvbench_compare.py` reports.
