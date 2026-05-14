---
description: |
  Tour and orientation for the C Parallel Library (`c/` directory) — stable C ABI exposing CCCL's
  parallel algorithms for FFI consumers. Covers dir layout, public API surface, the JIT-backed
  wrapper pattern, test layout, and the experimental STF sublibrary.
  Triggers: "what is cccl c", "c parallel library", "cccl c bindings", "cccl ffi", "c api".
---

# C Parallel Library

The C Parallel Library is the stable-ABI C face of CCCL's parallel primitives. It ships as
`cccl.c.parallel`, a shared library that Python (`cuda.compute`), Numba, and other language
runtimes load via FFI. All headers require the caller to `#define CCCL_C_EXPERIMENTAL` — the
entire surface is explicitly experimental and subject to change.

## Directory layout

```
c/
├── CMakeLists.txt              — enables parallel/ and experimental/stf/ subprojects
├── parallel/
│   ├── CMakeLists.txt          — builds cccl.c.parallel shared library
│   ├── include/cccl/c/        — public C headers (one per algorithm)
│   ├── src/                   — CUDA/C++ implementation (one .cu per algorithm)
│   │   ├── util/              — shared context, error, type, tuning utilities
│   │   ├── nvrtc/             — NVRTC / nvJitLink helpers
│   │   ├── jit_templates/     — JIT type-wrapper template system (see below)
│   │   └── hostjit/           — optional LLVM-backed host JIT (optional, ~20 min build)
│   ├── test/                  — CTest-based C++ tests (one per algorithm)
│   │   └── freestanding/      — header-isolation + bitcode tests
│   └── cmake/                 — CParallelHeaderTesting.cmake
└── experimental/stf/          — C bindings for the STF (stream task framework) backend
```

## Public API surface

Headers live under `c/parallel/include/cccl/c/`. One header per algorithm family:

| Header                   | Functions                                                          |
|--------------------------|---------------------------------------------------------------------|
| `types.h`                | `cccl_type_info`, `cccl_op_t`, `cccl_iterator_t`, `cccl_value_t`, enums |
| `reduce.h`               | `cccl_device_reduce_build[_ex]`, `cccl_device_reduce[_nondeterministic]`, `_cleanup` |
| `scan.h`                 | `cccl_device_scan_build[_ex]`, exclusive/inclusive scan variants, `_cleanup` |
| `for.h`                  | `cccl_device_for_build[_ex]`, `cccl_device_for`, `_cleanup`        |
| `transform.h`            | `cccl_device_transform_build[_ex]`, `cccl_device_transform`, `_cleanup` |
| `radix_sort.h`           | `cccl_device_radix_sort_build[_ex]`, sort variants, `_cleanup`     |
| `merge_sort.h`           | `cccl_device_merge_sort_build[_ex]`, sort variants, `_cleanup`     |
| `segmented_reduce.h`     | segmented reduce build/run/cleanup                                 |
| `segmented_sort.h`       | segmented sort build/run/cleanup                                   |
| `histogram.h`            | histogram build/run/cleanup                                        |
| `binary_search.h`        | lower/upper bound build/run/cleanup                                |
| `three_way_partition.h`  | three-way partition build/run/cleanup                              |
| `unique_by_key.h`        | unique-by-key build/run/cleanup                                    |

Every algorithm follows the same three-call pattern: `_build` (JIT-compiles a cubin for the
target SM), `_run` (launches the kernel), `_cleanup` (frees the cubin and library handle).
Extended `_build_ex` variants accept a `cccl_build_config` for extra compile flags and
include paths.

## Wrapper pattern

Each `.cu` in `src/` includes the corresponding CUB device algorithm and drives it through a
two-stage JIT pipeline:

1. `_build` calls use NVRTC + nvJitLink to compile a cubin specialized for the caller's
   `cccl_iterator_t` and `cccl_op_t` descriptors. Operators may be provided as LTO-IR blobs
   or as C++ source strings (`cccl_op_code_type`). The compiled cubin and `CUlibrary`/`CUkernel`
   handles are returned in the `_build_result_t` struct.
2. `_run` calls load the pre-built cubin and launch the kernel via the CUDA driver API.

The `jit_templates/` subsystem handles type-wrapper generation: it preprocesses C++ template
headers into embedded string literals (`jit_template_header_contents`) that NVRTC receives as
part of the compilation unit. This lets the C layer pass custom iterator and operator types
through to CUB without a C++ ABI dependency.

## Tests

`c/parallel/test/` holds one `test_<algorithm>.cpp` per algorithm. Tests link against
`cccl.c.parallel` and exercise the build/run/cleanup pattern from C++. The
`test/freestanding/` subdirectory tests header isolation (no C++ standard library linkage)
and bitcode paths.

Build with `-DCCCL_C_Parallel_ENABLE_TESTING=ON`. Header isolation tests use
`-DCCCL_C_Parallel_ENABLE_HEADER_TESTING=ON`.

## Experimental STF sublibrary

`c/experimental/stf/` exposes C bindings for CCCL's stream task framework. Enabled via
`-DCCCL_ENABLE_C_EXPERIMENTAL_STF=ON`. The single public header is
`include/cccl/c/experimental/stf/stf.h`. Tests under `stf/test/` cover tasks, logical data,
places, and CUDA kernels.

## Cross-references

- `cccl-python` — `cuda.compute` in `python/cuda_cccl/` is the primary consumer; it wraps
  every algorithm in this library via `_bindings.py` and `_cccl_interop.py`.
- `cccl-build` — build targets and preset flags for `cccl.c.parallel`.
- `cccl-test` — CTest targets for the test suite.

## Additional resources

- `references/tools.md` — build and test scripts for the C Parallel Library.
