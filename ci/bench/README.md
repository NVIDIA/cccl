# Benchmark Compare Scripts

This directory contains the scripts used by `.github/workflows/bench.yml` to compare benchmark results between two code states.

## Scripts

- `ci/bench/bench.sh`: CI-oriented wrapper that calls `ci/bench/compare_git_refs.sh`.
- `ci/bench/compare_git_refs.sh`: checks out `<base-ref>` and `<test-ref>` in temporary worktrees, then forwards all remaining args to `ci/bench/compare_paths.sh`.
- `ci/bench/compare_paths.sh`: configures/builds/runs CUB benchmarks and/or Python benchmarks in two source trees and runs comparison tools on produced JSON outputs.
- `ci/bench/parse_bench_matrix.sh`: parses `ci/bench.yaml` and emits a dispatch matrix JSON object for `.github/workflows/bench.yml`.

## Usage

Compare CUB benchmarks between two refs:

```bash
"./ci/bench/bench.sh" "origin/main" "HEAD" \
  --cub-filter "^cub\\.bench\\.copy\\.memcpy\\.base$"
```

Compare Python benchmarks between two refs:

```bash
"./ci/bench/bench.sh" "origin/main" "HEAD" \
  --python-filter "compute/reduce/sum\\.py"
```

Run both CUB and Python benchmarks:

```bash
"./ci/bench/bench.sh" \
  "origin/main" \
  "HEAD" \
  --arch "native" \
  --nvbench-args "..." \
  --cub-filter "^cub\\.bench\\.reduce\\..*$" \
  --python-filter "compute/reduce/sum\\.py"
```

Compare already checked-out trees:

```bash
"./ci/bench/compare_paths.sh" \
  "/path/to/base/cccl" \
  "/path/to/test/cccl" \
  --arch "native" \
  --cub-filter "^cub\\.bench\\.copy\\.memcpy\\.base$" \
  --python-filter "compute/transform/.*\\.py"
```

## Workflow Inputs

In `.github/workflows/bench.yml`:

- If `raw_args` is non-empty, it is parsed and passed directly to `ci/bench/bench.sh`.
- Otherwise, args are assembled from `base_ref`, `test_ref`, `arch`, `cub_filters`, `python_filters`, `nvbench_args`, `nvbench_compare_args`, and `nvbench_compare_legacy_args`.
- `nvbench_compare_args` are passed to `nvbench-compare-robust`; `nvbench_compare_legacy_args` are passed to diagnostic `nvbench-compare-legacy` runs.
- CUB filters are passed as `--cub-filter` flags. Python filters are passed as `--python-filter` flags.
- The workflow resolves NVBench `main` once and passes the resulting SHA to generated CUB build trees.
- Malformed quoted input (for example unmatched quotes) fails the workflow step.

## Python Benchmarks

Python benchmarks live under `python/cuda_cccl/benchmarks/` and use `cuda.bench` (the Python nvbench bindings). Each benchmark script outputs nvbench-compatible JSON with binary sidecars for bulk sample data.

For CUB-only comparisons, the caller's Python environment must provide the NVBench compare dependencies:

```bash
python3 -m pip install 'cuda-bench[compare]>=0.3.0'
```

For Python benchmarks, `compare_paths.sh`:

1. Creates isolated virtual environments for base and test trees.
2. Installs `cuda-cccl[bench-cuXX]` (editable, from each worktree), which pulls in `cuda-bench`, `cupy`, and all other benchmark dependencies.
3. Runs matching benchmark scripts in each venv.
4. Compares results using `${CCCL_BENCH_COMPARE_BIN}` when set, or `nvbench-compare-robust` from the benchmark venv otherwise. It emits robust `intervals`, `simple`, and `explain` reports, plus a diagnostic legacy report when `nvbench-compare-legacy` is available.

Python filters are regex patterns matched against relative paths under `python/cuda_cccl/benchmarks/`, for example:
- `compute/reduce/sum\.py` — single benchmark
- `compute/transform/.*\.py` — all transform benchmarks

## Artifacts

`compare_paths.sh` writes a run directory under `${CCCL_BENCH_ARTIFACT_ROOT:-$(pwd)/bench-artifacts}` containing:

- per-target JSON, binary sidecars, and markdown outputs for base/test runs,
- per-target robust compare reports (`compare/<target>.md`, `compare/<target>.simple.md`, and `compare/<target>.explain.md`) plus diagnostic legacy reports (`compare/<target>.legacy.md`) when available,
- grouped build logs (`build.base.log`, `build.test.log`), per-target run logs, and per-target compare logs,
- Python venv setup logs (`py.venv.base.log`, `py.venv.test.log`),
- `summary.md` with run metadata, per-target robust comparison summaries, and collapsible full compare reports.
