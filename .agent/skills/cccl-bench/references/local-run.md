# Local benchmark build and run

## Prerequisites

- Release build. CUB benchmarks fail CMake configuration in non-Release mode.
- `CMAKE_CUDA_ARCHITECTURES` must be set.
- A GPU must be available.

## Build

Use a CUB-enabled preset, e.g.:

```bash
cmake --preset cub-benchmarks   # or equivalent Release preset with CMAKE_CUDA_ARCHITECTURES
cmake --build build --target cub.bench.reduce.sum.base
```

All bench targets roll up under `cub.all.benches`. To build every benchmark:

```bash
cmake --build build --target cub.all.benches
```

Target naming: `cub.bench.<algo>.<variant>.base` for the baseline; `cub.bench.<algo>.<variant>.variant` when `CUB_ENABLE_TUNING=ON`.

## Run a single binary

```bash
./build/bin/cub.bench.reduce.sum.base \
  --stopping-criterion entropy \
  -d 0
```

Common nvbench flags:

| Flag                          | Meaning                                                          |
|-------------------------------|------------------------------------------------------------------|
| `-d 0`                        | Device index (required; nvbench breaks with multiple visible GPUs) |
| `--stopping-criterion entropy` | Adaptive stopping (recommended)                                  |
| `--timeout <s>`               | Per-state timeout                                                 |
| `--skip-time <s>`             | Skip states faster than this (noise floor)                        |
| `-a "Elements{io}=[16,20,24]"` | Override a runtime axis                                           |
| `--jsonbin result.json`       | Write results to JSON                                             |
| `--jsonlist-benches`          | Print benchmark metadata                                          |

## Compare two refs with bench.sh

`ci/bench/bench.sh` wraps `compare_git_refs.sh`. It checks out each ref in a temporary worktree, builds, runs, and compares. Run from the repo root inside the devcontainer (GPU required):

```bash
./ci/bench/bench.sh "origin/main" "HEAD" \
  --arch "native" \
  --cub-filter "^cub\.bench\.copy\.memcpy\.base$"
```

With Python filters:

```bash
./ci/bench/bench.sh "origin/main" "HEAD" \
  --python-filter "compute/reduce/sum\.py"
```

Artifacts land under `${CCCL_BENCH_ARTIFACT_ROOT:-./bench-artifacts}/`.

## Compare two already-checked-out trees

```bash
./ci/bench/compare_paths.sh \
  "/path/to/base/cccl" \
  "/path/to/test/cccl" \
  --arch "native" \
  --cub-filter "^cub\.bench\.copy\.memcpy\.base$"
```
