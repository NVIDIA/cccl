<!-- Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# cuda.coop.cutlass NVBench suite

This suite compares cooperative primitives inside CuTe DSL kernels with matched
C++ CUB kernels. NVBench, through `cuda.bench`, owns sampling, timing and output.
The kernels use the same inputs, group geometry, result ownership and algorithm
choices. Correctness checks run before performance measurement.

The default matrix contains 58 workloads:

| Family | Workloads | Representative coverage |
| --- | ---: | --- |
| Load and store | 8 | Direct/transpose, block, warp and logical warp |
| Reduce and sum | 6 | Integer/float sum, minimum, block and warp groups |
| Scan | 7 | Inclusive/exclusive sum and maximum |
| Batched reduction | 2 | Blocked and striped results |
| Exchange | 6 | Blocked/striped conversions at three group sizes |
| Shuffle | 4 | Up, down, offset and rotate |
| Merge sort | 6 | Keys/pairs at three group sizes |
| Histogram | 2 | Atomic and sort algorithms |
| TopK | 4 | Minimum/maximum keys and pairs |
| Radix operations | 6 | Keys, pairs and ranks in both directions |
| Run-length decode | 2 | A window and a complete output |
| Neighbors | 5 | Adjacent difference and discontinuity |

This is representative performance coverage across all collective families.
It is not every supported type, algorithm, partial tile or operator combination.
The `scan` dispatcher shares the measured scan implementations. Group queries,
storage ownership, payload construction and operator compilation support these
workloads; they are not separate timed collectives.

## Environment and source

From the repository root, activate a compatible CUDA 13/CuTe environment and
install the benchmark extra:

```bash
python -m pip install -e 'python/cuda_coop_cutlass[bench]'
export PYTHONPATH="$PWD/python/cuda_coop_cutlass${PYTHONPATH:+:$PYTHONPATH}"
```

The extra installs `cuda-bench[cu13,compare]`; it does not change the package's
CuTe compiler requirement. Use the same source checkout and compiler environment
for both providers. Record the commit, compiler/package versions, GPU and driver
with each result. A compiler candidate that is not publicly available remains an
explicit prerequisite for reproducing that candidate's results.

The C++ provider also requires `nvcc` and a compatible host C++ compiler. Linux
x86-64 and Linux ARM64 environments need native dependencies for their platform.
Availability of a Linux wheel does not establish Windows ARM64 support.

## Run and retain results

List the registered configurations first:

```bash
python python/cuda_coop_cutlass/benchmarks/run.py --list-cases
```

Run a bounded measurement for each provider on your reserved device:

```bash
python python/cuda_coop_cutlass/benchmarks/run.py --provider cutlass \
  --axis Blocks=1 --axis Launches=1 --min-samples 1 --min-time 0.01 \
  --timeout 5 --jsonbin cutlass.json --md cutlass.md \
  --artifacts cutlass-artifacts
python python/cuda_coop_cutlass/benchmarks/run.py --provider cub \
  --axis Blocks=1 --axis Launches=1 --min-samples 1 --min-time 0.01 \
  --timeout 5 --jsonbin cub.json --md cub.md --artifacts cub-artifacts
```

Use `--case` to select case names or glob patterns and the normal NVBench
`--axis` options to select launch geometry. Keep the NVBench JSON and its binary
sample sidecars together. Compiler/launch failures and failed correctness checks
must be treated as failed coverage, never as zero-time or successful skipped
measurements. Run one measurement process at a time on an otherwise idle GPU.

`--artifacts` retains generated source, final cubins, compiler/launch metadata
and the correctness receipt. Use a new directory for each run; existing result
directories are rejected. Without this option, compilation artifacts are
temporary and only the requested NVBench output files remain.

`Blocks` controls the number of independent thread blocks. Each kernel executes
one primitive per group, including its input loads and output stores. The suite
measures complete kernels, not isolated instruction latency. `Launches=1` uses a
direct launch; `Launches=16` captures sixteen launches in a graph. The reported
time is for the entire callback; divide by `Launches` for time per kernel.

The default `--timing cold` collects NVBench's cold samples. `--timing batch`
collects those samples plus a separate hot aggregate. NVBench flushes L2 before
each cold callback: a sixteen-launch graph gets one flush before the sequence,
not sixteen flushes. Compilation, module loading, allocation and correctness
checks happen outside timing. Both providers use the same CUDA Driver launch
path and NVBench's selected device and stream.

A single block and a large grid answer different questions. Increase the grid
until aggregate throughput stops improving before describing a saturated rate.
Repeat complete processes before interpreting small differences. Keep provider,
case, geometry, graph length and compiler settings matched, and retain all raw
samples. Performance collection is opt-in; no fixed slowdown gate is implied.

Compare matching provider runs with the standard NVBench tool:

```bash
nvbench-compare-robust --display intervals --no-color cub.json cutlass.json
```

The comparison uses cold sample distributions. Hot aggregates are separate
descriptive measurements. `AMBG` does not establish parity, and the comparator's
successful exit status does not mean every result is `SAME`. Preserve the
report, JSON, and relative binary sidecar directories together.

## Compare the standalone warp sum with CUB

Build the CUB target `cub.bench.reduce.warp_reduce_sum.base` and add the
build's `bin` directory to `PATH`. Run both benchmarks from the same checkout,
selecting the six shared types so NVBench can pair every result.

```bash
cub.bench.reduce.warp_reduce_sum.base -d 0 \
  -a 'T{ct}=[I8,I16,I32,I64,F32,F64]' --jsonbin cub-warp-sum.json
python python/cuda_coop_cutlass/benchmarks/reduce/warp_reduce_sum.py \
  -d 0 --jsonbin cute-warp-sum.json
nvbench-compare-robust cub-warp-sum.json cute-warp-sum.json
```

Each warp performs 128 dependent sums. Only lane zero carries the aggregate
into the next round. The other lanes supply their own signed inputs, keeping
the sum bounded even for I8. The measurement includes these input updates.
Earlier versions fed the return value back on every lane and measured a
different workload. Use the same recurrence on both sides of a comparison.

## Replay compiled kernels

Use `--compiled-kernels` to measure qualified cubins on another machine with the
same GPU architecture. Copy the complete source run's `--artifacts` directory,
including `qualification.json` and its `dDEVICE-CASE-bBLOCKS-l1` directories with
`kernel.json` and `kernel.cubin`. The source run must include `Launches=1` for
each selected case and grid size.

```bash
python python/cuda_coop_cutlass/benchmarks/run.py --provider cutlass \
  --compiled-kernels source-artifacts --blocks 1 4096 --launches 1 16 \
  --artifacts replay-artifacts --validate-only
```

The loader requires matching case, provider, grid, block shape, GPU architecture
and cubin hash. The runner regenerates inputs and independent references on the
target, checks direct launches, then checks poisoned outputs and guard regions
after graph replay. To measure, omit `--validate-only` and supply the usual
NVBench output and sampling options with a fresh artifact directory.

Replay uses the recorded cubins and requires no CuTe source compilation or CuTe
compiler installation on the target. The target still needs CUDA Python, NumPy
and NVBench. Its qualification receipt records runtime package versions; each
kernel retains the source `compilation_environment` and `replay_of` provenance.
Successful replay establishes execution of those binaries on the target, not
source-compilation support there.

## Compare two source trees in CI

The existing benchmark comparison runner supports this package:

```bash
./ci/bench/compare_paths.sh /path/to/base/cccl /path/to/test/cccl \
  --python-project cuda_coop_cutlass --python-filter '^run\.py$' \
  --nvbench-args '--provider cutlass --axis Blocks=1 --axis Launches=1'
```

Both trees must contain this suite and satisfy the unchanged compiler dependency.
The runner creates separate environments, installs each source tree's benchmark
extra, retains JSON/binary samples and emits NVBench comparison reports. The
workflow accepts the same arguments through its existing `raw_args` input.
It compares the selected provider across revisions; use separate direct runs
above when comparing CuTe with CUB at one revision.

## Baseline contracts

Histogram, TopK and Run Length Decode use the package's existing C++ adapter
text in the native baseline. Those adapters supply output projection, checked
wide counts and run preparation around CUB. TopK currently wraps CUB's internal
`cub::detail::block_topk`; CUB has no matching public BlockTopK class. These
comparisons hold the adapter work constant between C++ and CuTe. The archived
native source contains that complete text, and independent NumPy references
check results.

TopK output order is unspecified. Its cases use unique input keys and compare
selected key sets plus key/value associations, without imposing a device sort.
Radix rank checks stable ranks within the selected digit. Run Length Decode
checks both a window beginning inside a run and the complete expanded stream.
Neighbor operations check their edge ownership as well as interior values.

## Maintain the suite

Keep executable benchmark entry points separate from `_`-prefixed helpers.
When adding a primitive, supply a matching CUB operation and an independent host
reference, account for output ownership and temporary-storage synchronization,
and check the generated code when claimed work could be optimized away.
Unsupported cases need an explicit reason. Do not time compilation, allocation,
correctness checks or profiling as ordinary performance samples.
