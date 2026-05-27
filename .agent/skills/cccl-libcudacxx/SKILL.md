---
description: |
  Tour and orientation for the libcudacxx subdirectory — what the library is, how the
  include tree is laid out, the LLVM upstream-tracking model, CCCL-specific additions
  under `<cuda/std/...>`, test suite structure, and how style is enforced.
  Triggers: "what is libcudacxx", "how does libcudacxx work", "libcudacxx overview",
  "libcudacxx code style", "make this libcudacxx code compliant".
---

# cccl-libcudacxx

libcudacxx is CCCL's C++ standard library for both host and device code. It provides
`<cuda/std/...>` headers that mirror the C++ standard library (`<cuda/std/atomic>`,
`<cuda/std/tuple>`, etc.) and CCCL-specific extensions under `<cuda/...>`.

## Directory layout

| Path                            | Contents                                        |
|---------------------------------|--------------------------------------------------|
| `libcudacxx/include/cuda/std/`  | Headers ported or tracking LLVM libc++          |
| `libcudacxx/include/cuda/std/__cccl/` | CCCL config, prologue/epilogue machinery      |
| `libcudacxx/include/cuda/`      | CCCL-only extensions (not tracked upstream)     |
| `libcudacxx/test/`              | Lit test suite + unit tests                     |
| `libcudacxx/cmake/`             | CMake helpers used by the library's build       |

## Upstream-tracking model

`libcudacxx/include/cuda/std/` tracks LLVM libc++. Files ported from LLVM carry the
LLVM license. Files under `libcudacxx/include/cuda/` (CCCL-only) use the Apache License
v2.0 with LLVM Exceptions. License choice follows directory, not content.

When syncing from upstream, preserve LLVM naming and structure inside `cuda/std/`; apply
CCCL macros and visibility annotations on top without restructuring.

## CCCL-specific include subtree

`cuda/std/__cccl/` is the configuration layer. It is not included directly — `__config`
or the prologue/epilogue mechanism provides it. Every header must include:

1. `<cuda/std/detail/__config>` immediately after the include guard.
2. System-header pragmas (`_CCCL_IMPLICIT_SYSTEM_HEADER_*` guards).
3. `<cuda/std/__cccl/prologue.h>` as the last include before code.
4. `<cuda/std/__cccl/epilogue.h>` at the end of the file.

## Test suite

Tests live under `libcudacxx/test/`. Two categories:

- **Lit tests** — structured as `.pass.cpp` / `.compile.pass.cpp` / `.fail.cpp` /
  `.verify.cpp` / `.runfail.cpp` files; discovered and run by the lit runner. Test-authoring
  conventions, lit directives, and validation commands live in `references/style/testing.md`.
- **Unit tests** — conventional CMake/CTest targets; heavier integration scenarios.

Run lit tests via `cccl-test` (the underlying `build_and_test_targets.sh` script takes a `--lit-tests` flag).

## Style enforcement

Style is applied to `libcudacxx/include/` and `cudax/include/`. Pre-commit runs
clang-format and a set of custom checks. CI enforces the same set.

Style is split across focused references below. When making a file compliant, work
through naming → macros → templates → headers → visibility → comments/Doxygen in that
order, then verify with `pre-commit run --files <files>`.

## Additional resources

- `references/style/naming.md` — naming conventions: symbols, template params, plural collections, type qualification
- `references/style/macros.md` — `_CCCL_*` macro rules; API, host/device, nodiscard; C++ version policy and backports
- `references/style/templates.md` — template parameters, concepts, SFINAE, `constexpr`, trailing return types
- `references/style/headers.md` — include order, guard format, license selection
- `references/style/visibility.md` — `_CCCL_HIDE_FROM_ABI`, inlining, `noexcept` rules
- `references/style/doxygen.md` — inline comment policy and Doxygen requirements for public APIs
- `references/style/testing.md` — lit test layout, naming, test kinds, support headers, lit directives, validation commands
- `references/docs.md` — index of libcudacxx documentation (standard/extended/PTX API)
- `references/tools.md` — build and test scripts for libcudacxx
