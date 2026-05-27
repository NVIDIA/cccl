---
description: |
  Tour and orientation for the Thrust subdirectory — what the library is, how the
  include tree is laid out, the backend abstraction model, execution policies,
  relationship to CUB, and test suite structure.
  Triggers: "what is thrust", "thrust overview", "thrust algorithms",
  "thrust execution policies", "thrust backends".
---

# cccl-thrust

Thrust is CCCL's high-level parallel algorithms library. It provides an STL-like
interface (`thrust::sort`, `thrust::reduce`, `thrust::transform`, …) that runs on
multiple parallel backends — CUDA GPU, OpenMP, TBB, and serial CPU — selected at
call time via execution policies.

## Directory layout

| Path                      | Contents                                                  |
|---------------------------|-----------------------------------------------------------|
| `thrust/thrust/`          | Public headers — one file per algorithm                  |
| `thrust/thrust/system/`   | Backend implementations (cuda, cpp, omp, tbb)            |
| `thrust/thrust/detail/`   | Per-algorithm `.inl` dispatch internals                  |
| `thrust/thrust/iterator/` | Iterator adaptors (transform, zip, counting, …)          |
| `thrust/thrust/mr/`       | Memory resource layer                                    |
| `thrust/testing/`         | Main test suite (`.cu` files, CTest-driven)              |
| `thrust/testing/cuda/`    | CUDA-backend-specific tests                              |
| `thrust/examples/`        | Standalone `.cu` examples                                |
| `thrust/cmake/`           | CMake helpers: target lists, multi-config, header testing |

## Public API surface

Users include flat top-level headers: `<thrust/sort.h>`, `<thrust/reduce.h>`,
`<thrust/transform.h>`, etc. Each header's implementation body lives in
`thrust/detail/<algorithm>.inl`, included from the top-level header. There is no
`thrust/thrust.h` umbrella; users include only what they use.

Container types — `thrust::device_vector`, `thrust::host_vector`,
`thrust::universal_vector` — live in their own top-level headers.

## Backend abstraction

Each backend occupies `thrust/thrust/system/<name>/`:

| Backend | Namespace         | Description                                  |
|---------|-------------------|----------------------------------------------|
| `cuda`  | `thrust::cuda_cub` | GPU execution via CUDA + CUB device primitives |
| `cpp`   | `thrust::cpp`     | Serial CPU execution (STL / standard algorithms) |
| `omp`   | `thrust::omp`     | OpenMP parallel CPU                          |
| `tbb`   | `thrust::tbb`     | Intel TBB parallel CPU                       |

Each backend directory contains `execution_policy.h`, `detail/<algorithm>.h`, and
supporting headers. Algorithm dispatch follows C++ ADL: the execution policy type
selects the backend's overload set.

## Execution policies

Execution policies are the user-facing dispatch mechanism. Pass one as the first
argument to any Thrust algorithm to select a backend:

```
thrust::sort(thrust::device, v.begin(), v.end());   // CUDA backend
thrust::sort(thrust::host,   v.begin(), v.end());   // default host backend
thrust::sort(thrust::seq,    v.begin(), v.end());   // serial (no parallelism)

// stream-scoped:
thrust::sort(thrust::cuda::par.on(stream), v.begin(), v.end());
```

`thrust::device` and `thrust::host` resolve to the backends selected by
`THRUST_DEVICE_SYSTEM` and `THRUST_HOST_SYSTEM` macros (defaults: CUDA and CPP).
`thrust::cuda::par` is the concrete CUDA policy and supports `.on(stream)` for
stream association.

See `references/execution-policies.md` for the full policy type hierarchy and
stream/allocator extensions.

## CUB relationship

The CUDA backend (`thrust/system/cuda/detail/`) delegates device-side work to CUB
device-scope primitives. For example, `thrust::sort` calls `cub::DeviceRadixSort`
or `cub::DeviceMergeSort`; `thrust::reduce` calls `cub::DeviceReduce`. Thrust owns
the policy dispatch and host-side coordination; CUB owns the GPU kernel implementation.
This means CUDA-backend performance is directly tied to the corresponding CUB primitive.

## Test suite

`thrust/testing/` contains one `.cu` file per algorithm or feature. Tests are
CUDA-compiled and run via CTest. Backend-specific tests live in subdirectories:

- `testing/cuda/` — CUDA-backend tests (stream, CDP, memcpy flags, etc.)
- `testing/omp/` and `testing/cpp/` — host-backend variants

Catch2-based tests use the `catch2_test_*.cu` prefix. Legacy tests use a custom
`unittest` harness also in `testing/`.

Build and run with `cccl-build` / `cccl-test`.

## Additional resources

- `references/execution-policies.md` — policy type hierarchy, `.on(stream)`, allocator policies, custom backends
- `references/docs.md` — index of Thrust documentation (API reference, developer overview).
- `references/tools.md` — build and test scripts for Thrust.
