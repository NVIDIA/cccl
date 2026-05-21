---
name: libcudacxx-test
description: Write, update, and validate libcudacxx tests under libcudacxx/test.
---

# libcudacxx Test

## Organization

- Put CUDA Standard Library tests under `libcudacxx/test/libcudacxx/std/...`.
- Put CUDA-specific API tests under `libcudacxx/test/libcudacxx/cuda/...`, unless an adjacent `std/...` directory is clearly the established home for the functionality.
- Read nearby tests first and mirror their directory layout, file names, helper types, includes, and lit gates.

## Purpose

- Validate libcudacxx functionality. It is fundamental to verify:

  - Edge cases.
  - Input and output types.
  - Exception behavior.
  - Runtime and constant-evaluation behavior.
  - Device and host behavior.

## Test kinds

- `.pass.cpp`: compiles, links, runs, and returns 0.
- `.compile.pass.cpp`: compiles correctly.
- `.fail.cpp`: must fail compilation. Prefer precise `expected-error`, `expected-warning`, `expected-note`, or `expected-no-diagnostics` annotations when clang verify is supported.
- `.runfail.cpp`: compiles and runs but must return non-zero.

## Test structure

- All tests must have the correct license banner.
  
- Always include top level headers, never internal ones with `__` prefix.
- Include support headers `"test_macros.h"`, `"test_iterators.h"`, `"test_comparisons.h"`, when needed.

- Use `static_assert(...)` for compile-time guarantees and constexpr coverage.
- Use `<cuda/std/cassert>` and `assert(...)` for runtime checks.

- The `main` function must be present, dispatch runtime and static-evaluation tests, and return 0.
  
## Style

- Use `cuda::std` names, not `std::` names, unless the test is intentionally checking interoperability with host standard library types.
- Do not fully qualify names in header includes unless the test is intentionally checking interoperability with host standard library types.
- Mark helper functions that may run on host and device with `TEST_FUNC`; use `TEST_DEVICE_FUNC` for device-only helpers.
- `const`-qualification is discouraged.
- Don't use `noexcept` for helper functions unless strictly necessary.
- Do not use lambda expressions in host/device test code unless nearby tests already prove the pattern is supported.

## Portability

- Guard host-only or device-only behavior with `NV_IF_TARGET(NV_IS_HOST, (...))` and `NV_IF_TARGET(NV_IS_DEVICE, (...))` respectively.

- Use `TEST_STD_VER`, `TEST_COMPILER`, `TEST_CUDA_COMPILER`, `TEST_HAS_EXCEPTIONS`, and `TEST_THROW` from `"test_macros.h"` instead of spelling compiler or dialect probes directly.

- Unsupported platforms can be disabled with `UNSUPPORTED: <feature-name>` or `XFAIL: <feature-name>` lit directives. Some common feature names are `nvrtc`, `enable-tile`, `pre-sm-70`, `c++17`, `c++20`, `msvc`, `gcc-<version>`, or `clang-<version>`.

  - Always motivate unsupported features with a comment.

## Lit directives

- Put lit directives near the top of the file before includes.
- Common directives: `UNSUPPORTED:`, `XFAIL:`, `REQUIRES:`, `ADDITIONAL_COMPILE_DEFINITIONS:`, `ADDITIONAL_COMPILE_OPTIONS_HOST:`, `ADDITIONAL_COMPILE_OPTIONS_CUDA:`, `MODULES_DEFINES:`, and `CONSTEXPR_STEPS:`.
- For diagnostics in `.fail.cpp`, annotate the exact line that should fail when possible:

```cpp
bad_expression(); // expected-error {{message fragment}}
```

- Prefer checking the intended diagnostic over accepting any compile failure.

## Validation

Use targeted libcudacxx lit runs. Paths passed to `--lit-precompile-tests` and `--lit-tests` are relative to `libcudacxx/test/libcudacxx/`.

```bash
ci/util/build_and_test_targets.sh \
  --preset libcudacxx \
  --lit-precompile-tests "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp" \
  --lit-tests "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp"
```

If running lit directly, use the configured site file:

```bash
LIBCUDACXX_SITE_CONFIG=/home/fbusato/git_repo/cccl/build/<preset>/libcudacxx/test/libcudacxx/lit.site.cfg \
  lit -v libcudacxx/test/libcudacxx/<relative-test-path>
```

- Use `-Dexecutor=NoopExecutor()` for precompile-only validation when runtime execution is unavailable or GPU coverage is not required.
