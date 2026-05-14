# Template, concept, and `constexpr` conventions

## Template parameters

- Use `CamelCase` prefixed with `_` for all template parameters: `_Tp`, `_Up`, `_Key`,
  `_Value`, `_Alloc`.
- Avoid single-letter names. `_T` and `_U` are prohibited; `_Tp` and `_Up` are the
  conventional replacements.

## `constexpr` functions

Use `constexpr` for all functions that do not depend on run-time features (raw pointers,
device memory, system calls). The default should be `constexpr`; opt out only when a
specific run-time dependency prevents it.

## Trailing return types

When the return type is not explicitly spelled out (`auto`), a trailing return type is
strongly preferred:

```cpp
auto abs(float x) -> float;
auto make_pair(_Tp t, _Up u) -> pair<_Tp, _Up>;
```

This makes the return type visible at the point of declaration without requiring the
reader to parse the full signature.

## SFINAE and `enable_if`

Prefer `_CCCL_REQUIRES` or concept-based constraints when available. When falling back to
`enable_if`, place it in the trailing return type or as a defaulted non-type template
parameter — never in the function parameter list.

## Modern C++

The repository supports C++17 as the minimum, but many C++20 and later features have been
backported via CCCL macros and `cuda/std` wrappers. Prefer the backported form over
manual SFINAE when available:

- Concepts → `_CCCL_TEMPLATE(...)` / `_CCCL_REQUIRES(...)` / `_CCCL_CONCEPT`
- `consteval` replacements → `_CCCL_CONSTEVAL`
- `[[nodiscard]]` → use directly (supported from C++17)

Consult `libcudacxx/include/cuda/std/__cccl/` for the available portability macros before
writing bespoke SFINAE.
