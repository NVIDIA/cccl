---
name: libcudacxx-style
description: Make the code in libcudacxx/include, cudax/include compliant with the coding style
---

# libcudacxx Style

## Naming style

- Macros: macro style, e.g. `MY_MACRO`.
- Template parameters: CamelCase, e.g. `MyParameter`.
- All other symbols: snake style, e.g. `my_variable`.

All non-public symbols must be C++ reserved identifiers:

- `_` for macros and template parameters, e.g. `_MY_MACRO`., `_MyParameter`.
- `__` for all other symbols, e.g. `__my_variable`.

- Avoid single letter names for template parameters. Wrong: `_T`, correct: `_Tp`.

## Variables

- All variables that are not modified must use `const`. This includes variables initialized by casts (`static_cast`, `reinterpret_cast`, `bit_cast`), function return values, and loop-invariant computations.
- All variables that can be evaluated at compile-time must use `constexpr`.
- Consider using plural names for array, span, list, e.g. `int values[4]` instead of `int value[4]`.

## Function

Declaration/Definition:

- All functions must be marked `_CCCL_HOST_API`, `_CCCL_DEVICE_API`, or `_CCCL_API`.
- Non-template, non-`constexpr` functions must use `inline`.
- Most functions with a non-void return type shall use `[[nodiscard]]`. Exceptions are functions with known side effects, e.g. `cuda::std::copy`
- All functions that don't throw exception must use `noexcept`
- `constexpr` must be used for all functions that don't depend on run-time features, e.g. pointers.
- If the return type is not explicit (`auto`), then a trailing return type is strongly preferred, e.g. `auto abs(float) -> float`

Function call:

- All calls to free functions must be fully qualified starting from the global namespace, e.g. `::cuda::ceil_div`. This includes calls to functions defined in the same namespace, e.g. inside `cuda::`, call `::cuda::ceil_div(...)`, not `ceil_div(...)`. This does not apply to (static) member functions of classes.

## Types

- Type names must be fully qualified, except when they are already declared in the current namespace.
- This includes standard integer type aliases (`::cuda::std::size_t`, `::cuda::std::uintptr_t`, `::cuda::std::int32_t`, etc.) and any other `cuda::std` or standard library types. A local `using` declaration (e.g. `using ::cuda::std::size_t;`) is acceptable to avoid repetition within a function body.

## Headers

- All header inclusions must use the syntax `<header>`.
- Files must include all headers related to the symbols that they are using.
- No transitive header inclusion are allowed.
- Unneeded headers must be removed.
- The headers must be the most precise one, e.g. `#include <cuda/std/__type_traits/is_array.h>`.
- Headers in `cuda/std/__cccl/` must not be included directly (they are provided by `__config` or the prologue/epilogue mechanism).

- All headers must have the correct license. If the file is ported from LLVM libc++ then we *must* use the LLVM license.
- All headers must have the include guard, with the correct name: uppercase full path from the root, separated by `_`.
- The closing `#endif` always carries a comment repeating the guard name.
- Right after the include guard, the code must include:
```cpp
#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
```
- The last included header must be `#include <cuda/std/__cccl/prologue.h>` before the code, and `#include <cuda/std/__cccl/epilogue.h>` at the end of a file.

## Comments

- Commented code without a description is not allowed.
- Use Doxygen-style `//! @brief comments`.
- When a function is documented with Doxygen, it must include: `//! @brief`, `//! @param[in/out/in,out]` for every parameter, and `//! @return` for non-void functions.
- The `@brief/@param/@return` description must accurately reflect the current functionality of the function.

## General guidelines

- The code must reuse `cuda/` or `cuda/std` functionalities as much as possible, including macros.
- Try to use modern C++ as much as possible. The repository supports C++17 but many more recent functionalities have been backported with functions and macros.

## Prevent compiler errors and improve compatibility

- Never allow lambda expressions in device-only or host-device code.
- Protect host-only code with `#if !_CCCL_COMPILER(NVRTC)`.
- Remove unused code, variables, functions, types, template parameters, headers, etc.
- Variables that are unsigned, or that can become unsigned after template instantiation, must not check for negative values directly. Use `cuda::std::is_unsigned_v<T> ? false : (var < 0)` instead.
