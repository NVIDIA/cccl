# libcudacxx Style Guidance

Use this reference for `libcudacxx/include/**/*` and `cudax/include/**/*`.

## Naming Style

- Never use reserved keywords as variables, parameters, or function names, such as:
  `__in`, `__out`, `__inout`, `__input`, `__output`

## Functions

- Use `constexpr` for functions that do not depend on run-time features, such as pointers.
- If the return type is not explicit (`auto`), then a trailing return type is strongly preferred.

## Headers

- Use the correct license:
  - `libcudacxx/include/cuda/std` files ported from LLVM libc++ use the LLVM license.
  - `libcudacxx/include/cuda/` files use Apache License v2.0 with LLVM Exceptions.

## Comments

- Documented functions must include `//! @brief`, `//! @param[in/out/in,out]` for every parameter, and `//! @return` for non-void functions.
- The `@brief/@param/@return` description must accurately reflect the current functionality of the function.

## Compiler Compatibility

- Do not use lambda expressions in device-only or host-device code.
- Do not rely on deduction guides for initialization; use explicit template arguments instead.
