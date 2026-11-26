# fp benchmark tooling

Scripts for the nvbench benchmarks of the `cuda/fp` component, which live in
[`../bench/fp/`](../bench/fp/README.md): building and running them on demand, joining their
output into one table per benchmark, and comparing a run with a baseline for regression
tracking. Nothing here runs in CI; the benchmarks are built there and their SASS is compared,
the timings are for a developer's GPU.

| file | role |
|---|---|
| `fp_bench.py` | the driver: configure, build, run, table, compare; one call does it all |
| `fp_table.py` | joins the nvbench CSVs of the fpmp executables into `fpmp.csv` (one row per cell) and adds the SASS instruction count of each cell from the executable |
| `fp_compare.py` | compares an `fpmp.csv` with a baseline, cell by cell, and gives a verdict |
| `tests/` | `pytest -q libcudacxx/benchmarks/fp/tests` |

## Quick start

```bash
# all fp benchmarks: configure the libcudacxx-benchmark preset, build, run, table
python3 libcudacxx/benchmarks/fp/fp_bench.py

# one benchmark, two of its executables, on a chosen GPU
python3 libcudacxx/benchmarks/fp/fp_bench.py --bench fpmp --family arith --family trig --gpu GPU-<uuid>
```

The benchmarks are built in the preset's own tree, `build/${CCCL_BUILD_INFIX}/libcudacxx-benchmark`
under the checkout: the tree a manual `cmake --preset libcudacxx-benchmark` or
`ci/util/build_and_test_targets.sh` uses, so nothing is built twice, and it is reused on the next
run (`--build-dir` puts it elsewhere; an existing tree keeps its architectures unless `--arch`
names others). The output directory (`--out`, default `build/fp_bench`) holds the configure and
build logs and one directory per benchmark, `fpmp/`, with the nvbench csv/json/md/log of each
executable, the joined `fpmp.csv` and `fpmp.md`, and after a comparison `compare.csv` and
`compare.md`. One fpmp family takes 7 to 20 seconds to run; all eight about two minutes, plus the
build.

## Regression tracking

The question a change to the component has to answer is "did any cell get slower or less
accurate than on `main`?". `--compare-ref` answers it in one run:

```bash
git fetch origin main
python3 libcudacxx/benchmarks/fp/fp_bench.py --bench fpmp --compare-ref origin/main
```

This builds and runs the benchmarks of this tree, then adds a git worktree of `origin/main` under
`<out>/ref/src`, overlays the current `libcudacxx/benchmarks` directory onto it (both sides run
the same benchmark sources and tooling, each against its own library headers; the ref needs no
benchmarks of its own), builds it in its own `build/${CCCL_BUILD_INFIX}/libcudacxx-benchmark`
inside the worktree, runs it, and compares cell by cell:

```
[WARN] div fp32mp2 mid: SASS 42 -> 40 (the code changed)
[NOTE] div fp32mp2 mid: clk/ev 212.0 -> 180.0 (-15.1%, faster)
[OK]   compare: 306 cells against fpmp.csv (1 warn, 1 note)
```

The rules (the defaults of `fp_compare.py`): accuracy FAIL when `u2` grew by more than 1.5× and
by more than 0.5 u² (for the cells without `u2`, when `Bits` dropped); throughput FAIL when
`ev/clk/SM` dropped by more than 25 %, WARN beyond 10 %, the same for `clk/ev` the other way
round; any SASS count change is a WARN; 8 registers more is a WARN; the `Check` column of the run
itself is reported first and a FAIL there fails the run. `compare.md` is the detail table, one
row per cell with both values. The worktree is kept and moved to the ref on the next run, so the
second comparison costs one build of whatever changed.

Two other ways to get a baseline:

```bash
# compare with the table of an earlier run (any fpmp.csv)
python3 libcudacxx/benchmarks/fp/fp_bench.py --bench fpmp --baseline /some/run/fpmp/fpmp.csv

# keep per-GPU baselines: <dir>/<Device slug>.csv, chosen by the Device column of the results
python3 libcudacxx/benchmarks/fp/fp_bench.py --bench fpmp --baseline-dir ~/fp-baselines --store-baseline --note 'main 0d00df600b'
python3 libcudacxx/benchmarks/fp/fp_bench.py --bench fpmp --baseline-dir ~/fp-baselines
```

The exit status is 1 on any FAIL. A single run on an idle GPU moves by about 5 %; a GPU shared
with another process reads 20 to 40 % low, which the driver cannot tell from a regression:
check `nvidia-smi` first, or turn the performance rules into warnings with
`--compare-args '--perf-fail 100'`.

## Storage

The compile of one family opens a few thousand headers of the checkout and the build writes
tens of megabytes. On a host whose home directory is a network share with a 20 ms round trip,
that turned the 54 s build of a family into 7 minutes and a one-minute configure into eleven.
Keep the checkout and the build on local or scratch storage: a git worktree of the branch there
(`git -C ~/cccl worktree add /scratch/cccl dev/<branch>`), run the driver from it, and give it
`--out` and `--build-dir` on the same volume; `--compare-ref` then places its worktree under
`--out` as well. A checkout shared between hosts over NFS needs one build tree per host, since a
tree records the host's cmake and nvcc paths: set `CCCL_BUILD_INFIX` (the hostname, say) or pass
`--build-dir` and `--out` per host; the driver drops and reconfigures a tree that another host's
cmake made. The driver measures the open latency of the checkout's headers and the write
latency of the output directories when it starts: a `[NOTE]` with the figure above 3 ms per
open, a `[WARN]` above 8 ms per open or 20 ms per write. Measured points: 1.6 ms gave the
one-minute family build, 5.8 ms about three minutes, 10 ms seven.

## Options

| option | default | |
|---|---|---|
| `--bench NAME` | all | which benchmark(s); `fpmp` today |
| `--family NAME` | all `.cu` of the benchmark | which executable(s) |
| `--out DIR` | `build/fp_bench` | results, logs and the ref worktree |
| `--build-dir DIR` | `build/${CCCL_BUILD_INFIX}/libcudacxx-benchmark` | the CMake tree of this checkout; the ref worktree always builds in its own |
| `--arch LIST` | `native` for a new tree | `CMAKE_CUDA_ARCHITECTURES`; an existing tree keeps its own unless this is given |
| `--gpu ID` | all visible, nvbench runs on device 0 | `CUDA_VISIBLE_DEVICES`; a UUID is safest, the CUDA index orders by speed |
| `--jobs N` | 4 | parallel compile jobs (each takes a few GB) |
| `--nvbench-args` | `--min-time 0.1 --max-noise 5 --timeout 10` | passed to every executable; 5 % sample noise keeps a GPU with a swinging boost clock from sampling for minutes, the compared means are far tighter |
| `--cmake`, `--nvcc`, `--cpm-cache` | PATH, `/usr/local/cuda`, `$CPM_SOURCE_CACHE` | the toolchain; the CPM cache saves the nvbench download |
| `--no-build` | | run the executables already in the build tree |
| `--compare-args` | | extra options of the comparison script (`--perf-fail`, `--acc-factor`, ...) |

## Adding a benchmark

A new benchmark `<name>` is a directory `libcudacxx/benchmarks/bench/fp/<name>/` with one `.cu`
per executable (targets `libcudacxx.bench.fp.<name>.<file>.base`, see the fpmp CMake rules in
`libcudacxx/benchmarks/CMakeLists.txt`) and an entry in the `BENCHMARKS` registry of
`fp_bench.py`: the source directory, a marker file, the target prefix, and the table and compare
scripts. The scripts take the nvbench CSVs of the executables and the executables themselves
(for the SASS counts) and produce one `.csv`/`.md` table per benchmark; `fp_table.py` and
`fp_compare.py` are the fpmp ones and a model for the next.
