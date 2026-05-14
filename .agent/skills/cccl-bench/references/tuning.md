# CUB kernel parameter tuning

## How it works

Tunable benchmarks annotate kernel policy parameters with `%RANGE%` comments. CMake parses these at configure time to build a `.variant` target alongside the `.base` target. The `cccl.bench` Python harness (`benchmarks/scripts/cccl/bench/`) sweeps the parameter space, scores each variant against the base, and stores results in a SQLite database.

## Annotating a benchmark for tuning

In `base.cuh` or the `.cu` file, add `%RANGE%` comments above the tuning `#define`s:

```cpp
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1
```

Format: `// %RANGE% <DEFINE> <short_label> <start>:<end>:<step>`

The benchmark must guard tuning-specific code with `#if !TUNE_BASE` / `#endif` so the `.base` target compiles without the tuning parameters.

## Building with tuning enabled

```bash
cmake -DCUB_ENABLE_TUNING=ON ...
cmake --build build --target cub.bench.reduce.sum.variant
```

When `CUB_ENABLE_TUNING=ON`, CMake generates `<build>/cub.bench.reduce.sum.variant.h`. The harness rewrites this header for each parameter combination and rebuilds.

## Running the tuning harness

The harness lives in `benchmarks/scripts/cccl/bench/` and is invoked via `benchmarks/scripts/run.py` (or `search.py` for search-driven tuning). Run from the build directory:

```bash
cd <build>
python3 /path/to/benchmarks/scripts/run.py \
  -R "^cub\.bench\.reduce\.sum$"
```

Key flags:

| Flag                           | Meaning                                                |
|--------------------------------|--------------------------------------------------------|
| `-R <regex>`                   | Select benchmarks by name                              |
| `-a "Axis=Value"`              | Pin a runtime axis (e.g. `-a "Elements{io}=1048576"`) |
| `--num-shards N --run-shard K` | Parallel sharding                                      |
| `-P0`                          | Run P0 (priority 0) subset                             |
| `-l`                           | List available benchmarks and their variant counts     |

The harness builds `.base` once, then iterates `.variant` targets. Results go into a SQLite database (`cccl_bench.db` by default) keyed by `(ctk, cccl, gpu, variant)`.

## Scoring

Each variant is scored as a weighted sum of speedups over the base across the runtime axis space. Weights are computed by `benchmarks/scripts/cccl/bench/score.py`. The highest-scoring variant for each `(ctk, cccl, gpu)` combination is the tuning winner.

## Interpreting results

`benchmarks/scripts/analyze.py` reads the SQLite database and produces summary tables. `benchmarks/scripts/compare.py` compares two runs (e.g. before and after a tuning change).
