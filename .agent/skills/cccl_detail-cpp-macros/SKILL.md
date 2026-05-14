---
description: |
  CCCL `_CCCL_*` internal macro catalog — compiler detection, CUDA compiler
  and version queries, C++ dialect, execution-space qualifiers, visibility/ABI,
  diagnostics push/pop, and portability shims. Used by anyone authoring CCCL
  headers across libcudacxx, CUB, Thrust, and cudax.
  Triggers: "_CCCL_ macro", "_CCCL_HOST_DEVICE", "_CCCL_API", "compiler detection macro",
  "visibility macro", "diagnostic suppression", "_CCCL_STD_VER".
---

All `_CCCL_*` macros are defined under
`libcudacxx/include/cuda/std/__cccl/`. Every header there is included
transitively by `<cuda/__cccl_config>`. The macros are shared across the
entire CCCL family — they are not libcudacxx-private.

## Header map

| Header               | Contents                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------------|
| `compiler.h`         | Host compiler detection, CUDA compiler detection, compilation-phase predicates, `_CCCL_PRAGMA` |
| `dialect.h`          | `_CCCL_STD_VER`, dialect-gated `constexpr`/`consteval`, feature predicates                    |
| `execution_space.h`  | `_CCCL_HOST`, `_CCCL_DEVICE`, `_CCCL_HOST_DEVICE`, `_CCCL_GRID_CONSTANT`                     |
| `visibility.h`       | `_CCCL_API`, `_CCCL_HIDE_FROM_ABI`, `_CCCL_KERNEL_ATTRIBUTES` and variants                   |
| `attributes.h`       | `_CCCL_NODEBUG`, `_CCCL_ARTIFICIAL`, `_CCCL_PURE`, `_CCCL_ASSUME`, `_CCCL_NO_UNIQUE_ADDRESS`, etc. |
| `diagnostic.h`       | `_CCCL_DIAG_PUSH/POP`, per-compiler `_CCCL_DIAG_SUPPRESS_*`, `_CCCL_SUPPRESS_DEPRECATED_PUSH/POP` |
| `cuda_capabilities.h` | `_CCCL_PTX_ARCH`, RDC/EWP/CDP/PDL feature predicates, `_CCCL_LAUNCH_BOUNDS`                  |
| `deprecated.h`       | `CCCL_DEPRECATED`, `CCCL_DEPRECATED_BECAUSE`, dialect-gated deprecation shims               |

## Compiler detection

`_CCCL_COMPILER(ID)` and `_CCCL_COMPILER(ID, OP, MAJOR[, MINOR])` query the
host compiler. `_CCCL_CUDA_COMPILER(ID[, ...])` queries the CUDA compiler.
Both use a versioned dispatch pattern — each ID function-macro returns a
`(major, minor)` pair or `_CCCL_VERSION_INVALID()`.

Supported host IDs: `GCC`, `CLANG`, `MSVC`, `MSVC2019`, `MSVC2022`,
`MSVC2026`, `NVHPC`, `NVRTC`.

Supported CUDA compiler IDs: `NVCC`, `NVHPC`, `CLANG`, `NVRTC`.

Compilation-phase predicates:

| Macro                       | True when                                       |
|------------------------------|--------------------------------------------------|
| `_CCCL_CUDA_COMPILATION()`  | Compiling a `.cu` file                          |
| `_CCCL_HOST_COMPILATION()`  | `__CUDA_ARCH__` not defined                     |
| `_CCCL_DEVICE_COMPILATION()` | CUDA device pass active                         |
| `_CCCL_CUDACC()`            | Returns `(major, minor)` of the active CUDA toolkit |
| `_CCCL_CUDACC_AT_LEAST(M[, N])` | CUDA toolkit ≥ M.N                          |
| `_CCCL_CUDACC_BELOW(M[, N])` | CUDA toolkit < M.N                             |

See `references/compiler-detection.md` for usage patterns and version
comparison idioms.

## C++ dialect

`_CCCL_STD_VER` — integer year (2011, 2014, 2017, 2020, 2023, 2024).
Compare with `>=` / `<=`.

Dialect-gated `constexpr`: `_CCCL_CONSTEXPR_CXX20`, `_CCCL_CONSTEXPR_CXX23`.
Use these instead of raw `constexpr` when backporting to C++17.

Feature predicates (return 0/1): `_CCCL_HAS_CONCEPTS()`,
`_CCCL_HAS_PACK_INDEXING()`, `_CCCL_HAS_CHAR8_T()`,
`_CCCL_HAS_LONG_DOUBLE()`, `_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()`.

## Execution-space qualifiers

| Macro                   | Expands to                                                 |
|-------------------------|-------------------------------------------------------------|
| `_CCCL_HOST_DEVICE`     | `__host__ __device__` in CUDA builds; empty otherwise      |
| `_CCCL_HOST`            | `__host__` in CUDA builds                                  |
| `_CCCL_DEVICE`          | `__device__` in CUDA builds                                |
| `_CCCL_GRID_CONSTANT`   | `__grid_constant__` on supported toolchains (sm_70+, CUDA ≥ 12.8) |
| `_CCCL_EXEC_CHECK_DISABLE` | Disables NVCC exec-space-check for a function             |
| `_CCCL_LAUNCH_BOUNDS(...)` | `__launch_bounds__(...)` unless RDC is active             |

Always use `_CCCL_HOST_DEVICE` for any function callable from both host and
device contexts. Never use raw `__host__ __device__` in CCCL headers.

## Visibility and ABI

Use the function-qualifier macros, not the raw visibility macros, for all
function declarations.

| Macro                   | Use for                                                  |
|-------------------------|----------------------------------------------------------|
| `_CCCL_API`             | Standard internal `__host__ __device__` function — hidden from ABI |
| `_CCCL_HOST_API`        | Host-only variant                                        |
| `_CCCL_DEVICE_API`      | Device-only variant                                      |
| `_CCCL_NODEBUG_API`     | Same as `_CCCL_API` + debugger-skip (`inline` implied)   |
| `_CCCL_TRIVIAL_API`     | Same as `_CCCL_NODEBUG_API` + force-inline; for dispatch/CPO glue |
| `_CCCL_PUBLIC_API`      | Visible across DSO boundary — use when address appears in a public type |
| `_CCCL_KERNEL_ATTRIBUTES` | `__global__ _CCCL_VISIBILITY_HIDDEN` for kernel definitions |
| `_CCCL_HIDE_FROM_ABI`   | Legacy; prefer `_CCCL_API` for new code                  |
| `_CCCL_FORCEINLINE`     | Force-inline without ABI hiding                          |

`_LIBCUDACXX_HIDE_FROM_ABI` is a compatibility alias for external projects;
do not use it in new CCCL code.

See `references/visibility-abi.md` for the full attribute composition and
NVHPC workaround notes.

## Diagnostics

Always bracket suppression with push/pop — never suppress without restoring.

Host-compiler diagnostics:

```cpp
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wshadow")   // no-op on other compilers
_CCCL_DIAG_SUPPRESS_GCC("-Wshadow")
_CCCL_DIAG_SUPPRESS_MSVC(4267)
// ... code ...
_CCCL_DIAG_POP
```

NVCC/NVRTC diagnostics (numeric codes):

```cpp
_CCCL_BEGIN_NV_DIAG_SUPPRESS(20012, 20013)
// ... code ...
_CCCL_END_NV_DIAG_SUPPRESS()
```

Compound shortcut: `_CCCL_SUPPRESS_DEPRECATED_PUSH` / `_CCCL_SUPPRESS_DEPRECATED_POP`
suppresses deprecation warnings across all supported compilers simultaneously.

See `references/diagnostics.md` for the full suppress-macro table and
common warning codes.

## Additional resources

- `references/compiler-detection.md` — version comparison idioms, full ID list, freestanding/NVRTC notes
- `references/visibility-abi.md` — attribute composition matrix, `_CCCL_API` vs `_CCCL_TRIVIAL_API` decision, NVHPC quirks
- `references/diagnostics.md` — per-compiler suppress macros, common warning codes, push/pop patterns
