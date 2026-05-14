---
description: "CCCL's benchmarking infrastructure — nvbench C++ benchmarks, Python `cuda.bench` bindings, `ci/bench.yaml` PR bench requests, and the `cccl.bench` tuning harness. Triggers: \"benchmark this PR\", \"write a benchmark\", \"request a bench run\", \"compare perf before/after\", \"tune kernel params\"."
---

# cccl-bench

Orientation for CCCL's benchmark infrastructure: where bench sources live, how to write them, how to run them locally, how to request CI bench comparisons, and how tuning works.

## Source layout

| Location                              | Contents                                                                              |
|---------------------------------------|----------------------------------------------------------------------------------------|
| `cub/benchmarks/bench/<algo>/`         | CUB C++ benchmarks (`.cu` per variant, shared `.cuh` base)                            |
| `python/cuda_cccl/benchmarks/`         | Python benchmarks using `cuda.bench`                                                  |
| `benchmarks/cmake/CCCLBenchmarkRegistry.cmake` | CMake helpers: `add_bench`, `register_cccl_benchmark`, `register_cccl_tuning` |
| `benchmarks/scripts/cccl/bench/`       | `cccl.bench` Python tuning harness                                                    |
| `ci/bench/`                           | CI compare scripts (`bench.sh`, `compare_git_refs.sh`, `compare_paths.sh`)            |
| `ci/bench.yaml`                       | PR bench-request config (edit to request; must match template to merge)               |
| `ci/bench.template.yaml`              | Reset target — `ci/bench.yaml` must match this before merging                         |
| `.github/workflows/bench.yml`          | Benchmark Compare workflow (triggered by `ci/bench.yaml` dispatch)                    |

## Writing a C++ benchmark

C++ benchmarks use [nvbench](https://github.com/NVIDIA/nvBench). The standard pattern: a shared `base.cuh` defines the benchmark function and `NVBENCH_BENCH_TYPES` registration; each `.cu` selects type axes and, optionally, tuning parameter ranges via `%RANGE%` annotations.

See `references/nvbench-template.md` for a minimal template and axis patterns.

## Writing a Python benchmark

Python benchmarks mirror C++ targets and use `cuda.bench` (the Python nvbench binding). Each script registers a benchmark function, declares axes, and calls `bench.run_all_benchmarks(sys.argv)`. Filters in `ci/bench.yaml` match relative paths under `python/cuda_cccl/benchmarks/`.

See `references/nvbench-template.md` for a Python example alongside the C++ one.

## Running benchmarks locally

CUB benchmarks require a Release build with `CMAKE_CUDA_ARCHITECTURES` set. Build target `cub.bench.<algo>.<variant>.base`, then run the binary directly with nvbench flags. The `ci/bench/bench.sh` wrapper handles two-ref comparisons from a single command.

See `references/local-run.md` for build preset, binary invocation, and `bench.sh` usage.

## Requesting a CI bench run

Edit `ci/bench.yaml`, add regex filters and a GPU, add `[bench-only]` to commit messages, push, and CI dispatches `.github/workflows/bench.yml`. Artifacts include per-target JSON, markdown summaries, and a `summary.md`.

Full edit-and-tag flow: `references/ci-bench-request.md`.

## Tuning

The `cccl.bench` Python harness (`benchmarks/scripts/cccl/bench/`) drives kernel parameter search. Tunable benchmarks annotate parameters with `// %RANGE% DEFINE label start:end:step`. CMake generates a `.variant` target alongside the `.base` target when `CUB_ENABLE_TUNING=ON`. The harness builds both, sweeps the parameter space, and scores each variant against the base.

See `references/tuning.md` for the full workflow.

## Reset before merge

`ci/bench.yaml` must match `ci/bench.template.yaml` exactly. CI branch protection fails the diff check; reset the file before the final merge.

## Additional resources

- `references/nvbench-template.md` — C++ and Python benchmark skeletons with axis patterns
- `references/ci-bench-request.md` — full `ci/bench.yaml` edit flow, GPU pool, and `[bench-only]` tag
- `references/local-run.md` — local build and run commands, `bench.sh` usage
- `references/tuning.md` — `%RANGE%` annotation, `CUB_ENABLE_TUNING`, harness invocation
- `references/docs.md` — index of benchmark documentation.
- `references/tools.md` — benchmark scripts with purpose and cross-references.
