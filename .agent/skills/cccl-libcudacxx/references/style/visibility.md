# Visibility, linkage, and ABI rules

## API macros (required on every function)

Every function in `libcudacxx/include/` must carry exactly one API macro:

| Macro               | Meaning         |
|---------------------|-----------------|
| `_CCCL_HOST_API`    | Host-only       |
| `_CCCL_DEVICE_API`  | Device-only     |
| `_CCCL_API`         | Host + device   |

These macros control symbol visibility and `__forceinline__` / `inline` expansion.

## `inline` on non-template, non-`constexpr` functions

Non-template, non-`constexpr` functions must be marked `inline` in addition to the API
macro. Without `inline`, multiple-definition errors arise in translation units that
include the same header.

## `noexcept`

All functions that do not throw must carry `noexcept`. Omitting it leaves the ABI surface
wider than necessary and disables optimizer paths.

## `[[nodiscard]]`

Apply `[[nodiscard]]` to every function with a non-void return type unless the function
has a well-known side effect (e.g. `cuda::std::copy`, `cuda::std::fill`). When in doubt,
annotate.

## `_CCCL_HIDE_FROM_ABI`

Internal helpers that must not appear in the public ABI are annotated with
`_CCCL_HIDE_FROM_ABI`. This attribute combines `__attribute__((visibility("hidden")))` on
GCC/Clang with MSVC equivalents. Apply it to:

- Implementation-detail free functions in anonymous namespaces or `__` prefixed helpers.
- Static member functions of detail classes.

Do not apply it to anything part of the public API.

## `constexpr` variables at namespace scope

All `constexpr` variables at namespace or global scope must use `inline` to avoid ODR
violations across translation units:

```cpp
inline constexpr bool __is_constant_evaluated_v = ...;

template <typename _Tp>
inline constexpr bool is_integral_v = ...;
```
