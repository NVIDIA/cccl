---
description: |
  Tour and orientation for the CUB subdirectory — what the library is, how the
  include tree is organized across block/warp/device/agent layers, test suite
  layout and naming, the tuning policy mechanism, and how CUB integrates with
  the CCCL buildsystem.
  Triggers: "what is cub", "cub overview", "cub primitives", "cub block scan",
  "cub device reduce", "cub tuning policy".
---

# cccl-cub

CUB is CCCL's CUDA primitive library. It provides cooperative algorithms at
three hardware scopes — thread block, warp, and full device — plus internal
agent-level building blocks used to compose device-wide algorithms.

## Directory layout

| Path                              | Contents                                                             |
|-----------------------------------|----------------------------------------------------------------------|
| `cub/cub/block/`                  | Block-level cooperative primitives (`BlockReduce`, `BlockScan`, `BlockSort`, …) |
| `cub/cub/warp/`                   | Warp-level primitives (`WarpReduce`, `WarpScan`, `WarpExchange`, …) |
| `cub/cub/device/`                 | Device-wide dispatch facades (`DeviceReduce`, `DeviceScan`, `DeviceSort`, …) |
| `cub/cub/device/dispatch/`        | Dispatch layer: one `dispatch_*.cuh` per algorithm                   |
| `cub/cub/device/dispatch/tuning/` | Tuning policy structs, one `tuning_*.cuh` per algorithm              |
| `cub/cub/agent/`                  | Internal multi-block agents — not part of the public API             |
| `cub/cub/iterator/`               | Iterator adapters (cache-modified, transform, etc.)                  |
| `cub/cub/thread/`                 | Single-thread primitives (thread reduce, scan, load/store)           |
| `cub/cub/grid/`                   | Grid-scope utilities (even-share, mapping, queue)                    |
| `cub/cub/detail/`                 | Internal helpers — not part of the public API                        |
| `cub/cmake/`                      | CMake helpers: `CubUtilities.cmake`, `CubHeaderTesting.cmake`, etc.  |
| `cub/test/`                       | Catch2 and legacy CTest test suite                                   |
| `cub/benchmarks/`                 | Performance benchmarks (enabled via `CCCL_ENABLE_BENCHMARKS`)        |
| `cub/examples/`                   | Usage examples under `block/` and `device/` subdirs                  |

## Header conventions

The umbrella include is `<cub/cub.cuh>`. It is not usable from NVRTC — NVRTC
callers must include specific headers (e.g., `<cub/device/device_reduce.cuh>`).
Every header follows the same structure:

1. `#pragma once`
2. `#include <cub/config.cuh>` — pulls in namespace macros and compiler config.
3. System-header pragmas (`_CCCL_IMPLICIT_SYSTEM_HEADER_*` guards).
4. Implementation includes, then `CUB_NAMESPACE_BEGIN` / `CUB_NAMESPACE_END`.

Prefer `<cub/device/device_X.cuh>` for device algorithms; the block headers for
kernel code. The `agent/` and `detail/` subtrees are internal — treat anything
not under `block/`, `warp/`, `device/`, `iterator/`, or `thread/` as unstable.

## Tuning policy mechanism

Every device algorithm has a matching `tuning_*.cuh` file under
`cub/cub/device/dispatch/tuning/`. Each file defines:

- A **policy struct** (e.g., `detail::reduce::reduce_policy`) aggregating
  per-kernel parameters: `threads_per_block`, `items_per_thread`, block
  algorithm, load modifier, vector size.
- A **policy selector** — a `constexpr` function or concept-constrained
  overload set that picks parameters based on `compute_capability`, accumulator
  type, and offset type.

The dispatch layer (`dispatch_*.cuh`) queries the selector at compile time and
instantiates the agent template. Users can inject a custom policy hub by passing
it as a template argument to the `Dispatch*` type — tests named
`*_custom_policy_hub.cu` exercise this path.

See `references/tuning-policies.md` for the full struct shapes and how to
author a custom policy hub.

## Test suite layout

Tests live in `cub/test/`. Two naming conventions:

- `catch2_test_<scope>_<algorithm>[_variant].cu` — Catch2-based; the dominant
  new style. Discovered automatically by `GLOB_RECURSE`.
- `test_<name>.cu` — legacy style; uses a bespoke test harness with
  `CUB_DEBUG_SYNC` enabled.

**`%PARAM%` variant expansion.** Tests that need to cover multiple compile-time
configurations embed directives of the form:

```
// %PARAM% TEST_DIM_X dimx 1:7:32:65:128
```

`cccl_parse_variant_params` reads these at CMake configure time and generates
one CTest target per combination. Target names take the form
`cub.test.<scope>.<algorithm>.<label>` (e.g.,
`cub.test.block.reduce.dimx_32.dimyz_1`).

**`_fail.cu` tests** compile-test for expected diagnostic errors; they use
`cccl_add_xfail_compile_target_test` and carry `expected-error` regex markers.

**`_api.cu` tests** exercise the public API surface with `--extended-lambda`
enabled.

Run targeted builds and tests via `cccl-build` and `cccl-test`.

## Buildsystem integration

CUB is built as part of CCCL via the `cub` preset family. When included from the
CCCL superproject, `CCCL_ENABLE_CUB` controls whether the full `cub/CMakeLists.txt`
is processed. Key CMake options:

| Option                    | Default | Effect                                |
|---------------------------|---------|---------------------------------------|
| `CUB_ENABLE_HEADER_TESTING` | ON      | Compile-test all public headers       |
| `CUB_ENABLE_TESTING`      | ON      | Build `cub/test/` targets             |
| `CUB_ENABLE_EXAMPLES`     | ON      | Build `cub/examples/` targets         |
| `CUB_ENABLE_TUNING`       | OFF     | Build tuning-exploration benchmarks   |

CTest targets follow `cub.test.<scope>.<algorithm>[.<variant>]`. The
`cub.compiler_interface` target carries compiler flags; all test targets link it.

## Additional resources

- `references/tuning-policies.md` — policy struct shapes, per-algorithm selector
  pattern, and how to author a custom policy hub
- `references/docs.md` — index of CUB documentation (API reference, developer overview, tuning).
- `references/tools.md` — build and test scripts for CUB.
