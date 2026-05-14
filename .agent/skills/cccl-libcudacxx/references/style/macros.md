# Macro rules

## API linkage macros

Every function must carry exactly one of:

| Macro                 | Use                  |
|-----------------------|----------------------|
| `_CCCL_HOST_API`      | Host-only function   |
| `_CCCL_DEVICE_API`    | Device-only function |
| `_CCCL_API`           | Host-device function |

## `inline` requirement

Non-template, non-`constexpr` functions must use `inline`.

## `[[nodiscard]]`

Most functions with a non-void return type must use `[[nodiscard]]`. Exceptions: functions
with known side effects (e.g. `cuda::std::copy`).

## `noexcept`

All functions that do not throw must be marked `noexcept`.

## `constexpr`

Use `constexpr` for all functions that do not depend on run-time features (pointers,
device memory, etc.). Variables that can be evaluated at compile time must also be
`constexpr`.

## `inline` on `constexpr` variables

All `constexpr` variables at namespace or global scope must use `inline`, including
template variables:

```cpp
inline constexpr int foo = 42;
template <typename T>
inline constexpr bool is_foo_v = ...;
```

## `const` on non-modified variables

All variables that are not modified must use `const`, including:
- Variables initialized by casts (`static_cast`, `reinterpret_cast`, `bit_cast`)
- Function return values captured in a local
- Loop-invariant computations

## Uniform initialization

Use uniform initialization for class constructors and compile-time conversions (not
enforced for builtin types):

```cpp
constexpr auto x = int{sizeof(float)};
```

## Compiler-compatibility macros

- Never allow lambda expressions in device-only or host-device code.
- Do not rely on deduction guides for initialization; use explicit template arguments.
- Protect host-only code with `#if !_CCCL_COMPILER(NVRTC)`.
- Variables that are unsigned (or can become unsigned after template instantiation) must
  not check for negative values directly:

```cpp
// Wrong
if (var < 0) { ... }

// Correct
if (::cuda::std::is_unsigned_v<T> ? false : (var < 0)) { ... }
```

## General macro policy

- Reuse `cuda/` or `cuda/std/` macros wherever they exist; do not roll bespoke macros for
  things already covered by the CCCL or standard library infrastructure.
- Remove unused macros, variables, functions, types, template parameters, and headers.
