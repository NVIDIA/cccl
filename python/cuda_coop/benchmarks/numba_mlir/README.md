# Device-side warp reduction benchmarks: cuda.coop (numba-cuda-mlir) vs CUB

Measures `cuda.coop` warp sum, written in Python and compiled with
numba-cuda-mlir, against the CUB C++ benchmarks
`cub/benchmarks/bench/reduce/warp_reduce_sum.cu` (`WarpReduce::Reduce(x, plus)`)
and `warp_reduce_sum_method.cu` (`WarpReduce::Sum`). Both sides compile CUB from
the same checkout: cuda.coop through NVRTC, C++ through nvcc.

## Method

The design follows the numba-cuda device-side benchmark harness (NVIDIA/cccl
PR #6846), ported to numba-cuda-mlir, and mirrors
`nvbench_helper/device_side_benchmark.cuh`:

- No global memory traffic. Each thread seeds `data` from `%clock`
  (`random_value`), applies the operation `UnrollFactor` times in a dependent
  chain (`data = op(data)`), and stores the result only if `%smid` is
  `0xFFFFFFFF`, which never happens. The store keeps the chain alive.
- 256-thread blocks, `launch_bounds=256` (`.maxntid 256, 1, 1` on both sides).
- Grid = max active blocks per SM (`cuOccupancyMaxActiveBlocksPerMultiprocessor`
  on the compiled kernel) times the SM count, as in the C++ benchmark.
- `Unroll=loop` is a counted loop, the C++ `for`; `Unroll=full` writes the
  calls out, the C++ `cuda::static_for`.
- nvbench measures cold GPU time per launch (`exec_tag::gpu | no_batch` in
  C++, `state.exec(..., batched=False)` in Python) on nvbench's stream.

Compilation and warmup are outside the timed region: cuda.coop specializes a
kernel on its full launch configuration, grid included, so the benchmark
compiles and runs the exact timed launch once before `state.exec`. Host launch
cost is measured separately by `launch_overhead.py` and
`cpp/launch_overhead.cu`, since nvbench's cold GPU time excludes it.

Correctness comes first: `check_warp_reduce.py` checks every supported type
and algorithm against NumPy, and against a single-call CUB executable
(`cpp/check_warp_reduce.cu`) for bitwise agreement.

## Files

| File | Purpose |
|---|---|
| `device_side_benchmark.py` | `random_value`, `sreg_clock`, `sreg_smid` intrinsics, `sink`, kernel factory, occupancy grid |
| `bench_warp_reduce.py` | pynvbench benchmark `base`: axes `T{ct}`, `Algorithm{ct}` (`sum`, `reduce_add`), `UnrollFactor`, `Unroll` |
| `check_warp_reduce.py` | correctness check vs NumPy and the CUB checker |
| `audit_sass.py` | SASS, register and memory comparison against the C++ binaries |
| `launch_overhead.py`, `cpp/launch_overhead.cu` | host launch cost per call |
| `compare.py` | Python/C++ ratio table from repeated nvbench JSON runs |
| `cpp/check_warp_reduce.cu` | single-call CUB checker |

`Algorithm=sum` is `coop.sum(coop.this_warp(), x)`, which lowers to
`WarpReduce::Sum`. `Algorithm=reduce_add` passes a Python `add` callback to
`coop.reduce`, which reaches `WarpReduce::Reduce(x, op)` with a generic
operator. Passing `operator.add` instead normalizes to `Sum`.

## Running

From a cuda.coop checkout with numba-cuda-mlir and `cuda-bench` installed:

```bash
python check_warp_reduce.py
python bench_warp_reduce.py -a "Unroll=[loop,full]" --json python.json
python launch_overhead.py
```

C++ side, from the CCCL root (`-DCPM_nvbench_SOURCE` points at the nvbench
revision the `cuda-bench` wheel was built from):

```bash
cmake --preset cub-benchmark -DCMAKE_CUDA_ARCHITECTURES=120 -DCPM_nvbench_SOURCE=<nvbench>
ninja -C build/cub-benchmark cub.bench.reduce.warp_reduce_sum.base cub.bench.reduce.warp_reduce_sum_method.base
build/cub-benchmark/bin/cub.bench.reduce.warp_reduce_sum_method.base -a UnrollFactor=128 --json cpp-sum.json
build/cub-benchmark/bin/cub.bench.reduce.warp_reduce_sum.base --json cpp-reduce.json
python compare.py --python python.json --cpp-sum cpp-sum.json --cpp-reduce cpp-reduce.json
```

## Limitations

- float16 is skipped: cuda.coop warp `sum` and `reduce` reject it.
- The sink buffer is a kernel argument in Python and a `__device__` array in
  C++; this adds a constant 5 to 7 instruction prologue to the Python kernel.
- `warp_reduce_base.cuh` launches on the default stream rather than
  `launch.get_stream()`; `warp_reduce_sum_method.cu` uses nvbench's stream.
