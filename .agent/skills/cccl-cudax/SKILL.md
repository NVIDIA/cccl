---
description: |
  Tour and orientation for the cudax subdirectory — what CUDA Experimental is,
  the no-stability-guarantee contract, include tree layout, major feature areas
  (streams, containers, memory resources, execution, STF, places, graph,
  copy), test suite structure, and how features graduate to stable CCCL libraries.
  Triggers: "what is cudax", "cudax overview", "cudax experimental",
  "cuda::experimental", "cudax features".
---

# cccl-cudax

cudax (`cuda/experimental/`) is CCCL's staging ground for features under active
design. Everything in the `cuda::experimental::` namespace carries zero stability
guarantees — API and ABI can change or disappear without notice, at any cadence.
It is not shipped with the CUDA Toolkit; it is available only from the CCCL
GitHub repository. C++17 or newer is required; NVCC 12.3+ with GCC 7+, Clang 9+,
or MSVC 2019+ as host compiler.

## Stability contract

No stability guarantees whatsoever. Features live here while their design
solidifies and the community provides feedback. Once a feature is considered
ready, it graduates to a stable CCCL library (`libcudacxx`, CUB, or Thrust) with
a stable API. There is no documented timeline or graduation checklist; graduation
happens on maintainer judgment.

The CMake target is `cudax::cudax`, exposed only when `CCCL_ENABLE_UNSTABLE` is
set before `find_package` or `add_subdirectory`.

## Include tree

Public headers live under `cudax/include/cuda/experimental/`. Each feature area
has a top-level `.cuh` entry point and a `__<area>/` detail directory.

| Entry header        | Feature area                                                         |
|---------------------|----------------------------------------------------------------------|
| `container.cuh`     | `uninitialized_buffer`, `graph_buffer`                               |
| `stream.cuh`        | `stream`, `stream_ref`                                               |
| `memory_resource.cuh` | Stream-ordered memory resources, `graph_resource`                    |
| `execution.cuh`     | stdexec-based async execution model                                  |
| `launch.cuh`        | Typed kernel launch parameters                                       |
| `graph.cuh`         | CUDA graph construction and management                               |
| `places.cuh`        | Execution/data affinity across multi-device systems                  |
| `stf.cuh`           | Sequential Task Flow (STF) programming model                         |
| `copy.cuh`          | Typed async copy                                                     |
| `copy_bytes.cuh`    | Byte-wise mdspan host↔device copy                                    |
| `green_context.cuh` | SM-partitioned green contexts (CUDA 12.4+)                           |
| `group.cuh`         | Cooperative group abstractions with mappings                         |
| `kernel.cuh`        | Kernel attribute introspection (`kernel_ref`)                        |
| `library.cuh`       | Library context handle (`library_ref`)                               |
| `cufile.cuh`        | cuFile integration (CUDA 12.9+, Linux only)                          |

## Feature areas

| Area               | Key types / entry point                              | Notes                                                                                                            |
|--------------------|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Containers         | `uninitialized_buffer<T, Props...>`                  | Owning device storage, memory location encoded in properties                                                    |
| Streams            | `stream`, `stream_ref`                               | Owning and non-owning `cudaStream_t` wrappers with RAII                                                         |
| Memory resources   | stream-ordered `async_resource`, `graph_resource`   | Compatible with `libcudacxx` `cuda::mr::` framework                                                             |
| Execution          | P2300 (`std::execution`) senders/schedulers          | `stream_context`, `sync_wait`, `bulk`, `when_all`; needs `-allow-unsupported-compiler`                          |
| Launch             | `launch<Config>(kernel, args...)`                    | Typed launch with compile-time grid/block encoding                                                              |
| Graph              | graph capture + node ops, `graph_buffer`            | Graph-lifetime allocations                                                                                       |
| Places             | `exec_place`, `data_place`                           | Execution/data affinity across devices, green contexts, stream pools, multi-device grids; standalone (no STF required) |
| STF                | Sequential Task Flow                                | Task-graph model with auto dependency tracking; large subproject, `cudax_ENABLE_CUDASTF`                        |
| Copy / copy_bytes  | `copy`, typed mdspan transfer                        | Byte-wise and typed async copies, relaxed and strict ordering                                                   |
| Green contexts     | `green_context_helper`                               | SM-partitioned sub-device contexts; CUDA 12.4+                                                                  |
| Group              | cooperative algorithms + mappings                    | Group synchronizers, segmented algorithms                                                                        |
| cuFile             | GDS integration                                      | Linux only; requires CUDA 12.9+                                                                                 |

## Header conventions

Mirrors `libcudacxx`. Public entry headers are `include/cuda/experimental/<name>.cuh`.
Implementation detail headers live in `__<name>/` subdirectories and use the same
`_CCCL_IMPLICIT_SYSTEM_HEADER_*` pragma guards. Include the top-level entry header
only; never include detail headers directly.

## Test suite

Tests use Catch2 under `cudax/test/`. CTest targets are `cudax.test.<area>`,
mirroring the include tree (e.g. `cudax.test.execution`, `cudax.test.containers`).
STF and Places have separate CMake subdirectories (`test/stf/`, `test/places/`)
with distinct build requirements. cuFile tests are conditional on
`cudax_ENABLE_CUFILE` and require CUDA 12.9+ on Linux.

## Cross-references

- Build and run cudax targets → `cccl-build`, `cccl-test`
- Stable stdlib layer that cudax features graduate into → `cccl-libcudacxx`

## Additional resources

- `references/docs.md` — index of cudax documentation (STF, memory resources, API reference).
- `references/tools.md` — build and test scripts for cudax.
