---
grand_parent: Standard API
parent: Numerics Library
nav_order: 2
---

# `<cuda/std/complex>`

## Omissions

When using libcu++ with NVCC, `complex` does not support `long double` or
  `complex` literals (`_i`, `_if`, and `_il`).
NVCC warns on any usage of `long double` in device code, because `long double`
  will be demoted to `double` in device code.
This warning can be suppressed silenced with `#pragma`s, but only globally, not
  just when using `complex`.
User-defined floating-point literals must be specified in terms of
  `long double`, so they lead to warns that are unable to be suppressed.

## Customizations

Our implementation by default recovers infinite values during multiplication and division. This adds a significant runtime overhead, so we allow disabling that canonicalization if it is not desired.

Definition of `LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS` disables canonicalization for both multiplication *and* division.

Definition of `LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_DIVISION` or `LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_DIVISION` disables canonicalization for multiplication or division individually.

