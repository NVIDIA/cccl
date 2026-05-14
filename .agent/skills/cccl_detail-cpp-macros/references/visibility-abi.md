# Visibility and ABI reference

## Function-qualifier macro matrix

| Macro                      | `__host__ __device__` | Visibility | `inline` | Force-inline | Debugger-skip |
|----------------------------|----------------------|-----------|----------|--------------|---------------|
| `_CCCL_API`                | `__host__ __device__` | hidden     | no       | no           | no            |
| `_CCCL_HOST_API`           | `__host__`            | hidden     | no       | no           | no            |
| `_CCCL_DEVICE_API`         | `__device__`          | hidden     | no       | no           | no            |
| `_CCCL_NODEBUG_API`        | `__host__ __device__` | hidden     | yes      | no           | yes           |
| `_CCCL_NODEBUG_HOST_API`   | `__host__`            | hidden     | yes      | no           | yes           |
| `_CCCL_NODEBUG_DEVICE_API` | `__device__`          | hidden     | yes      | no           | yes           |
| `_CCCL_TRIVIAL_API`        | `__host__ __device__` | hidden     | yes      | yes          | yes           |
| `_CCCL_TRIVIAL_HOST_API`   | `__host__`            | hidden     | yes      | yes          | yes           |
| `_CCCL_TRIVIAL_DEVICE_API` | `__device__`          | hidden     | yes      | yes          | yes           |
| `_CCCL_PUBLIC_API`         | `__host__ __device__` | default    | no       | no           | no            |
| `_CCCL_PUBLIC_HOST_API`    | `__host__`            | default    | no       | no           | no            |
| `_CCCL_PUBLIC_DEVICE_API`  | `__device__`          | default    | no       | no           | no            |

## Decision guide

- **Default choice for internal helpers:** `_CCCL_API`.
- **Trivial dispatchers / CPO glue:** `_CCCL_TRIVIAL_API`. These are always
  force-inlined and debuggers step through them transparently.
- **"Step-over" utilities** (`cuda::std::move`, `cuda::std::forward`): `_CCCL_NODEBUG_API`.
  Debuggers show the frame in a stacktrace but won't let you set the active
  frame to it.
- **Function whose address appears in a public type:** `_CCCL_PUBLIC_API`.
  GCC warns if a `hidden` function's address is embedded in a `default`-visibility type.
- **CDP-callable functions:** `_CCCL_CDP_API` — expands to `_CCCL_API` when CDP
  is available, `_CCCL_HOST_API` otherwise.

`_CCCL_API` and `_CCCL_FORCEINLINE` cannot be combined — `_CCCL_API` already
implies `inline` through `_CCCL_HIDE_FROM_ABI`. Use `_CCCL_TRIVIAL_API` instead.

## Raw visibility macros (avoid in new code)

| Macro                           | Expands to                                                        |
|---------------------------------|-------------------------------------------------------------------|
| `_CCCL_VISIBILITY_HIDDEN`       | `__attribute__((__visibility__("hidden")))`                       |
| `_CCCL_VISIBILITY_DEFAULT`      | `__attribute__((__visibility__("default")))`                      |
| `_CCCL_VISIBILITY_EXPORT`       | `__declspec(dllexport)` on MSVC; `_CCCL_VISIBILITY_DEFAULT` elsewhere |
| `_CCCL_TYPE_VISIBILITY_HIDDEN`  | Hidden type visibility (`__type_visibility__` if available)       |
| `_CCCL_TYPE_VISIBILITY_DEFAULT` | Default type visibility                                           |
| `_CCCL_FORCEINLINE`             | `__forceinline` (MSVC) / `__inline__ __attribute__((__always_inline__))` |
| `_CCCL_FORCEINLINE_LAMBDA`      | `__attribute__((__always_inline__))` on lambdas (non-MSVC)        |
| `_CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION` | `__attribute__((__exclude_from_explicit_instantiation__))` if available |
| `_CCCL_HIDE_FROM_ABI`           | `_CCCL_VISIBILITY_HIDDEN _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION inline` |

## NVHPC workaround

NVHPC has issues with visibility attributes on symbols with internal linkage.
All visibility macros degrade gracefully on NVHPC — `_CCCL_API` expands to
`_CCCL_HOST_DEVICE` alone.

## Kernel attributes

```cpp
_CCCL_KERNEL_ATTRIBUTES           // __global__ _CCCL_VISIBILITY_HIDDEN
_CCCL_LAUNCH_BOUNDS(N)            // __launch_bounds__(N) unless RDC active
_CCCL_LAUNCH_BOUNDS(N, M)         // __launch_bounds__(N, M) unless RDC active
_CCCL_GRID_CONSTANT               // __grid_constant__ for const kernel params (sm70+, CUDA 12.8+)
_CCCL_BLOCK_SIZE(NTID, NCTA)      // __block_size__ / __cluster_dims__ on hopper+ (CUDA 12.9+)
```

`_CCCL_KERNEL_ATTRIBUTES` can be redefined by users until CCCL 4.0 (for
backwards compat with `CUB_DETAIL_KERNEL_ATTRIBUTES`). Use
`_CCCL_KERNEL_ATTRIBUTES` on all kernel definitions — never raw `__global__`.

## `_CCCL_GLOBAL_VARIABLE`

Marks file-scope variables accessible from device code. Expands to `__device__`
during device compilation (except NVHPC), empty otherwise. Required for
non-builtin-type global variables referenced in device code.
