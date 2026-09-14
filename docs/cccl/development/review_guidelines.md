# CCCL Review Guidelines

Guidelines for reviewing CCCL changes, distilled from past regressions. Each guideline
states what a reviewer — human or AI — should flag in a diff.

## build.compiler-matrix (important, all C++ code)

<!-- provenance: #534→#536 MSVC int128/asm; #1403→#1423 VLAs; #1320→#1417 NVHPC/GCC; ICC visibility →#1152 -->

Flag compiler-specific constructs unless guarded by a feature macro: `__int128`, GNU inline
`asm`, `__attribute__`, VLAs, visibility attributes, one-compiler warning suppressions.
CCCL must build with GCC, Clang, MSVC, NVHPC/NVC++ (incl. `-stdpar`), and NVRTC; PR CI does
not cover all of these, so green CI is not sufficient. Compiler-detection refactors must
stay equivalent for every supported compiler — beware masquerading (NVHPC defines
`__GNUC__`, Intel defines Clang/GCC macros). Benchmarks and tests count too.

## build.cuda-mode-detection (critical, all C++ headers)

<!-- provenance: #1199→#1224 nvc++ -stdpar fully broken -->

Flag raw `__CUDACC__`, `__NVCOMPILER`, `__CUDA_ARCH__`, or other vendor macros outside the
CCCL config headers; use `_CCCL_CUDA_COMPILATION()`, `_CCCL_COMPILER(...)`,
`_CCCL_DEVICE_COMPILATION()`. NVC++ `-stdpar` enables CUDA without `__CUDACC__`.
Candidate for a pre-commit grep.

## api.vocabulary-type-refactor (critical, public types in thrust/libcudacxx/cub)

<!-- provenance: #262→#1249 pair trivial copyability; #454→#1286,#1425 complex reverted twice -->

When a public vocabulary type (`thrust::pair`/`tuple`/`complex`, …) is reimplemented,
re-derived, or aliased, verify preservation of: trivial copyability and layout (downstream
`memcpy`s them), size/alignment/ABI, implicit conversions and promotions, overload
resolution, numerical behavior. rmm/PyTorch/RAPIDS break first — confirm the third-party
smoke tests ran (commit tags can skip them).

## infra.pin-deps (important, CMake/CI/submodules)

<!-- provenance: #534 nvbench `#main` →#582 -->

Flag dependencies fetched by branch name (`CPMAddPackage("gh:org/repo#main")`, `GIT_TAG
main`); pin a commit or tag. Candidate for a pre-commit grep.

## Retired

<!-- provenance: #651, #1257 -->

- `if constexpr` fallback instantiation firing `static_assert`s pre-C++17: obsolete since
  C++17 is the minimum dialect.
