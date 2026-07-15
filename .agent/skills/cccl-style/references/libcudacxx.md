# libcudacxx Style Guidance

Use this reference for `libcudacxx/include/**/*` and `cudax/include/**/*`.

## Naming Style

All non-public symbols must be C++ reserved identifiers:

- `_` for macros and template parameters, e.g. `_MY_MACRO`, `_MyParameter`.
- `__` for all other symbols, e.g. `__my_variable`.
- Never use reserved keywords, such as `__in`, `__out`, or `__inout` as variables, parameters, or function names.
- Avoid single-letter template parameter names. Wrong: `_T`; correct: `_Tp`.

## Class / Struct

- Data member names have postfix `_`, e.g. `class __myclass { int __data_; };`.
- Constructor parameter names should match class/struct data member names without the postfix `_`, e.g. `class __myclass { __myclass(int __data) : __data_(__data) {} };`.

## Functions

- Use `constexpr` for functions that do not depend on run-time features, such as pointers.
- If the return type is not explicit (`auto`), then a trailing return type is strongly preferred.

## Headers

- Use the correct license:
  - `libcudacxx/include/cuda/std` files ported from LLVM libc++ use the LLVM license.
  - `libcudacxx/include/cuda/` files use Apache License v2.0 with LLVM Exceptions.
- Headers use include guards with names derived from the uppercase full path and closing `#endif` comments repeating the guard name.
- Right after the include guard, include:

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

- The last included header before code must be `<cuda/std/__cccl/prologue.h>`, and `<cuda/std/__cccl/epilogue.h>` must appear at the end of the file.

## Comments

- Use Doxygen-style `//! @brief` comments.
- Documented functions must include `//! @brief`, `//! @param[in/out/in,out]` for every parameter, and `//! @return` for non-void functions.
- The `@brief/@param/@return` description must accurately reflect the current functionality of the function.

## Compiler Compatibility

- Do not use lambda expressions in device-only or host-device code.
- Do not rely on deduction guides for initialization; use explicit template arguments instead.
