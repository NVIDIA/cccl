# Test suite organization

## Two test categories

| Category  | Location                       | Runner                       |
|-----------|--------------------------------|------------------------------|
| Lit tests | `libcudacxx/test/` (recursive) | `llvm-lit` / CCCL lit wrapper |
| Unit tests | `libcudacxx/test/` (CMake targets) | CTest                        |

Run lit tests via `cccl-test` (the underlying `build_and_test_targets.sh` script takes
a `--lit-tests` flag). Unit tests run as normal CTest targets.

## Lit test naming

Lit tests are plain `.cpp` files with a suffix that encodes expected outcome:

| Suffix       | Meaning                                                    |
|--------------|-------------------------------------------------------------|
| `.pass.cpp`  | Must compile and run successfully                           |
| `.fail.cpp`  | Must fail to compile                                        |
| `.verify.cpp` | Compiler output is checked against `// expected-*` annotations |

## Lit test layout

Tests mirror the include tree. A test for `<cuda/std/atomic>` lives under
`libcudacxx/test/std/atomics/`. CCCL-specific extensions under `<cuda/...>` have tests
under `libcudacxx/test/cuda/`.

## Lit test file structure

Each lit test is self-contained:

```cpp
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. ...
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{build}
// RUN: %{run}

#include <cuda/std/...>
// test body
```

The `RUN:` lines invoke the lit substitution variables defined by the test suite
configuration.

## Doxygen in production headers

Functions documented with Doxygen must include all three tags; partial documentation is
not allowed:

```cpp
//! @brief One-line description of what the function does.
//! @param[in] x Description of x.
//! @param[out] result Description of result.
//! @return What the return value means.
```

Omit the `//! @return` line only for `void` functions. The description must reflect
current behavior — stale docs are treated as bugs.

## Comments

- Commented-out code without an explanatory comment is not allowed.
- Use `//! @brief` for Doxygen; `//` for inline implementation notes.
