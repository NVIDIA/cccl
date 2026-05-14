# Compiler detection reference

## `_CCCL_COMPILER` and `_CCCL_CUDA_COMPILER` usage

Both macros use the same versioned dispatch pattern:

```cpp
_CCCL_COMPILER(ID)                     // true if host compiler is ID
_CCCL_COMPILER(ID, OP, MAJOR)          // compare major version
_CCCL_COMPILER(ID, OP, MAJOR, MINOR)   // compare major.minor version

_CCCL_CUDA_COMPILER(ID)                // true if CUDA compiler is ID
_CCCL_CUDA_COMPILER(ID, OP, MAJOR)
_CCCL_CUDA_COMPILER(ID, OP, MAJOR, MINOR)
```

`OP` is a C comparison operator: `<`, `<=`, `==`, `>=`, `>`.

Examples:

```cpp
#if _CCCL_COMPILER(GCC, >=, 12)
// GCC 12+
#endif

#if _CCCL_CUDA_COMPILER(NVCC, <, 12, 8)
// NVCC older than 12.8
#endif

#if _CCCL_COMPILER(CLANG) && _CCCL_CUDA_COMPILER(CLANG)
// clang-cuda
#endif
```

## Host compiler IDs

| ID          | Detected by             |
|-------------|-------------------------|
| `GCC`       | `__GNUC__` (not clang, not icc) |
| `CLANG`     | `__clang__`             |
| `MSVC`      | `_MSC_VER`              |
| `MSVC2019`  | MSVC 19.20–19.29        |
| `MSVC2022`  | MSVC 19.30–19.49        |
| `MSVC2026`  | MSVC 19.50+             |
| `NVHPC`     | `__NVCOMPILER`          |
| `NVRTC`     | `__CUDACC_RTC__`        |

Intel Classic (`icc`/`icpc`) is not supported — CCCL emits a warning.
NVRTC is treated as freestanding (no host stdlib).

## CUDA compiler IDs

| ID    | Detected by                       |
|-------|-----------------------------------|
| `NVCC` | `__NVCC__` (inside `.cu` compile) |
| `NVHPC` | `_NVHPC_CUDA`                     |
| `CLANG` | clang `__CUDA__` + `_CCCL_COMPILER(CLANG)` |
| `NVRTC` | `_CCCL_COMPILER(NVRTC)` (same compiler) |

## CUDA toolkit version (`_CCCL_CUDACC`)

`_CCCL_CUDACC()` returns `(major, minor)` of the active CUDA toolkit, or
`_CCCL_VERSION_INVALID()` when not in a CUDA compilation.

Use the shorthand predicates:

```cpp
_CCCL_CUDACC_AT_LEAST(12)        // toolkit >= 12.0
_CCCL_CUDACC_AT_LEAST(12, 5)     // toolkit >= 12.5
_CCCL_CUDACC_BELOW(13)           // toolkit < 13.0
_CCCL_CUDACC_EQUAL(12, 8)        // toolkit == 12.8
```

## Freestanding / NVRTC notes

`_CCCL_FREESTANDING()` is 1 when `_CCCL_ENABLE_FREESTANDING` is defined or
the compiler is NVRTC. `_CCCL_HOSTED()` is its complement.
`_CCCL_HOSTJIT()` is 1 in freestanding non-NVRTC contexts (GPU JIT with
host-stdlib access).

## Compilation-phase predicates

`_CCCL_CUDA_COMPILATION()`, `_CCCL_HOST_COMPILATION()`, and
`_CCCL_DEVICE_COMPILATION()` test the current compilation pass, not the
compiler. In a `.cu` file, both `_CCCL_CUDA_COMPILATION()` and (on the
host pass) `_CCCL_HOST_COMPILATION()` can be 1 simultaneously.

## `_CCCL_PRAGMA`

Portable pragma emission:

```cpp
_CCCL_PRAGMA(unroll 4)               // emits #pragma unroll 4 (or __pragma on MSVC)
_CCCL_PRAGMA_UNROLL(N)               // portable loop-unroll hint
_CCCL_PRAGMA_UNROLL_FULL()           // unroll all iterations
_CCCL_PRAGMA_NOUNROLL()              // prevent unrolling
```
