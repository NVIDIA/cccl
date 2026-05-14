# Thrust Execution Policy Reference

## Policy type hierarchy

```
thrust::execution_policy<Derived>
├── thrust::host_execution_policy<Derived>      — base for host backends
│   └── thrust::cpp::execution_policy<>         — serial CPU
│   └── thrust::omp::execution_policy<>         — OpenMP CPU
│   └── thrust::tbb::execution_policy<>         — TBB CPU
└── thrust::device_execution_policy<Derived>    — base for device backends
    └── thrust::cuda::execution_policy<Derived> — CUDA GPU
        └── thrust::cuda::par_t                 — concrete CUDA policy object
```

## Built-in policy objects

| Object              | Header                                        | Description                   |
|---------------------|-----------------------------------------------|-------------------------------|
| `thrust::seq`       | `<thrust/execution_policy.h>`                 | Serial; no parallelism        |
| `thrust::host`      | `<thrust/execution_policy.h>`                 | Host backend (macro-selected) |
| `thrust::device`    | `<thrust/execution_policy.h>`                 | Device backend (macro-selected) |
| `thrust::cuda::par` | `<thrust/system/cuda/execution_policy.h>`    | CUDA backend                  |
| `thrust::omp::par`  | `<thrust/system/omp/execution_policy.h>`     | OpenMP backend                |
| `thrust::tbb::par`  | `<thrust/system/tbb/execution_policy.h>`     | TBB backend                   |

## Stream association

```cpp
#include <thrust/system/cuda/execution_policy.h>

cudaStream_t stream;
cudaStreamCreate(&stream);

// Associate all work in the call with `stream`:
thrust::sort(thrust::cuda::par.on(stream), v.begin(), v.end());
```

`.on(stream)` returns a new policy object bound to the given CUDA stream. Algorithms
launched with this policy run asynchronously on that stream and obey standard CUDA
stream ordering.

## Allocator-aware policies

Combine a stream with a custom allocator for temporary storage:

```cpp
thrust::sort(thrust::cuda::par(my_allocator).on(stream), v.begin(), v.end());
```

Thrust uses the allocator for internal scratch buffers instead of `cudaMalloc`.
Common use: caching allocators (e.g., `thrust::mr::pool_resource`) to avoid repeated
device allocations in hot loops.

## Custom backends

Derive from `thrust::host_execution_policy<MyPolicy>` or
`thrust::device_execution_policy<MyPolicy>` and provide ADL-visible overloads for
the algorithms you specialise. Algorithms without an overload fall back to the
parent class's backend.

`thrust/examples/minimal_custom_backend.cu` — minimal working example.

## Backend selection macros

| Macro                    | Default                     | Effect                        |
|--------------------------|-----------------------------|-----------------------------|
| `THRUST_DEVICE_SYSTEM`   | `THRUST_DEVICE_SYSTEM_CUDA` | Sets `thrust::device` backend |
| `THRUST_HOST_SYSTEM`     | `THRUST_HOST_SYSTEM_CPP`    | Sets `thrust::host` backend   |

Backends: `CUDA`, `CPP`, `OMP`, `TBB`. Change at compile time; do not change at runtime.
