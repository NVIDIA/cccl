# Test suite organization

## Two test categories

| Category   | Location                       | Runner                        |
|------------|--------------------------------|-------------------------------|
| Lit tests  | `libcudacxx/test/` (recursive) | `llvm-lit` / CCCL lit wrapper |
| Unit tests | `libcudacxx/test/` (CMake targets) | CTest                     |

Run lit tests via `cccl-test` (the underlying `build_and_test_targets.sh` script takes a
`--lit-tests` flag). Unit tests run as normal CTest targets.

## Directory layout

- Put CUDA standard library tests under `libcudacxx/test/libcudacxx/std/...` — mirrors the
  include tree under `<cuda/std/>`.
- Put CCCL-specific API tests under `libcudacxx/test/libcudacxx/cuda/...`, unless an adjacent
  `std/...` directory is already the established home for the functionality.
- Read nearby tests first and mirror their layout, file names, helper types, includes, and
  lit gates.

## Test kinds — file-name suffixes

| Suffix             | Meaning                                                                   |
|--------------------|---------------------------------------------------------------------------|
| `.pass.cpp`        | Must compile, link, run, and return 0.                                    |
| `.compile.pass.cpp` | Must compile cleanly; not executed.                                      |
| `.fail.cpp`        | Must fail to compile.                                                     |
| `.verify.cpp`      | Compiler output checked against `// expected-*` annotations (clang verify). |
| `.runfail.cpp`     | Compiles and runs but must return non-zero.                               |

Prefer `.verify.cpp` with precise `expected-error` / `expected-warning` / `expected-note` /
`expected-no-diagnostics` annotations over plain `.fail.cpp` when clang verify is supported —
this catches the *right* failure, not just any failure.

## Test file structure

Each test is self-contained:

```cpp
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{build}
// RUN: %{run}

#include <cuda/std/...>
#include "test_macros.h"     // when needed

int main(int, char**) {
  // ... compile-time + runtime checks ...
  return 0;
}
```

Rules:

- Always include top-level headers — never internal `__`-prefixed headers.
- Include support headers when needed: `"test_macros.h"`, `"test_iterators.h"`,
  `"test_comparisons.h"`.
- Use `static_assert(...)` for compile-time guarantees and `constexpr` coverage.
- Use `<cuda/std/cassert>` and `assert(...)` for runtime checks.
- `main` must be present, dispatch runtime and constant-evaluated branches, and return 0.
- The `RUN:` lines invoke the lit substitution variables defined by the suite configuration.

## Test style

- Use `cuda::std::` names, not `std::`, unless the test is intentionally checking
  interoperability with host standard-library types.
- Do not fully qualify names in header includes unless the test is intentionally checking
  interoperability with host standard-library types.
- Mark helper functions that may run on host and device with `TEST_FUNC`; use
  `TEST_DEVICE_FUNC` for device-only helpers.
- `const`-qualification on test locals is discouraged.
- Do not mark helper functions `noexcept` unless strictly necessary.
- Do not use lambda expressions in host/device test code unless nearby tests already prove
  the pattern is supported on the target configuration.

## Portability

- Guard host-only or device-only behavior with `NV_IF_TARGET(NV_IS_HOST, (...))` or
  `NV_IF_TARGET(NV_IS_DEVICE, (...))`.
- Use `TEST_STD_VER`, `TEST_COMPILER`, `TEST_CUDA_COMPILER`, `TEST_HAS_EXCEPTIONS`, and
  `TEST_THROW` from `"test_macros.h"` instead of spelling out compiler or dialect probes
  directly.
- Unsupported platforms can be disabled with `UNSUPPORTED: <feature>` or `XFAIL: <feature>`
  lit directives. Common feature names: `nvrtc`, `enable-tile`, `pre-sm-70`, `c++17`,
  `c++20`, `msvc`, `gcc-<version>`, `clang-<version>`. Always motivate unsupported features
  with a comment.

## Lit directives

Place lit directives near the top of the file, before includes.

Common directives:

- `UNSUPPORTED: <feature>`
- `XFAIL: <feature>`
- `REQUIRES: <feature>`
- `ADDITIONAL_COMPILE_DEFINITIONS: <macro>`
- `ADDITIONAL_COMPILE_OPTIONS_HOST: <opt>`
- `ADDITIONAL_COMPILE_OPTIONS_CUDA: <opt>`
- `MODULES_DEFINES: <macro>`
- `CONSTEXPR_STEPS: <count>`

For diagnostics in `.fail.cpp` / `.verify.cpp`, annotate the exact line that should fail
when possible:

```cpp
bad_expression(); // expected-error {{message fragment}}
```

Prefer checking the intended diagnostic over accepting any compile failure.

## Validation commands

Targeted run via the test-and-build driver (paths are relative to
`libcudacxx/test/libcudacxx/`):

```bash
ci/util/build_and_test_targets.sh \
  --preset libcudacxx \
  --lit-precompile-tests "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp" \
  --lit-tests           "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp"
```

Direct `lit` invocation, when needed, uses the site config from the build directory:

```bash
LIBCUDACXX_SITE_CONFIG=<path-to-cccl>/build/<preset>/libcudacxx/test/libcudacxx/lit.site.cfg \
  lit -v libcudacxx/test/libcudacxx/<relative-test-path>
```

Use `-Dexecutor=NoopExecutor()` for precompile-only validation when runtime execution is
unavailable or GPU coverage is not required.

Comment / Doxygen conventions for production headers (and for code comments anywhere in the
library) live in `doxygen.md`.
