# nvbench benchmark templates

## C++ (nvbench)

Minimal structure: a shared `base.cuh` defines the benchmark function and registration macro; `.cu` files select type axes and optionally declare tuning ranges.

`base.cuh`:
```cpp
#pragma once

#include <cub/device/device_reduce.cuh>
#include <nvbench_helper.cuh>

template <typename T, typename OffsetT>
void my_algo(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto elements = state.get_int64("Elements{io}");

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch,
    [&](nvbench::launch& launch) {
      // invoke kernel here
    });
}

NVBENCH_BENCH_TYPES(my_algo, NVBENCH_TYPE_AXES(value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
```

`sum.cu` (variant selecting types; adding tuning ranges):
```cpp
#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

using value_types = all_types;
using op_t        = ::cuda::std::plus<>;
#include "base.cuh"
```

Axis suffix conventions:
- `{ct}` — compile-time axis (type parameter)
- `{io}` — runtime axis affecting I/O throughput display
- No suffix — plain runtime axis

Available type aliases (`nvbench_helper.cuh`): `all_types`, `value_types`, `offset_types`, `integral_types`, `float_types`.

Tuning annotations (`%RANGE%`):
```
// %RANGE% DEFINE_NAME short_label start:end:step
```
CMake parses these to build `cub.<prefix>.<algo>.variant` alongside the `.base` target.

## Python (`cuda.bench`)

Python benchmarks mirror C++ targets. Filters in `ci/bench.yaml` match relative paths under `python/cuda_cccl/benchmarks/`.

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import SIGNED_TYPES as TYPE_MAP, as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import OpKind, make_reduce_into


def bench_my_algo(state: bench.State):
    type_str  = state.get_string("T{ct}")
    dtype     = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements{io}"))

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        d_in  = generate_data_with_entropy(num_items, dtype, "1.000", alloc_stream)
        d_out = cp.empty(1, dtype=dtype)

    state.add_element_count(num_items)
    state.add_global_memory_reads(num_items * d_in.dtype.itemsize, "Size")
    state.add_global_memory_writes(d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        # invoke op here via launch.get_stream()
        pass

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_my_algo)
    b.set_name("base")
    b.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    bench.run_all_benchmarks(sys.argv)
```

Python benchmarks output nvbench-compatible JSON consumed by the same `nvbench-compare` tool used for C++.
