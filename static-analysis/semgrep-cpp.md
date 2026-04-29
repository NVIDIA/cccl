# Semgrep C/C++ Analysis

**Tool:** semgrep with vendored C++ and CUDA rules
**Total findings:** 928 (608 from text report display)
**Scan scope:** C/C++/CUDA source and header files

## Summary

Semgrep uses pattern-based matching with 47 C++ rules and 11 CUDA rules to detect unsafe functions, memory management issues, race conditions, type safety problems, and CUDA-specific bugs.

## Findings by Rule

| Rule | Count | Description |
|------|-------|-------------|
| reinterpret-cast | 95 | Use of `reinterpret_cast` |
| c-style-pointer-cast | 47 | C-style casts `(type*)ptr` |
| const-cast | 33 | Use of `const_cast` |
| raw-malloc | 3 | Direct `malloc` usage (prefer smart pointers) |
| using-namespace-std | 1 | `using namespace std` in source |

## Key Observations

- **Cast-heavy codebase:** 175 findings are cast-related — expected in a low-level CUDA library that interfaces with C APIs and hardware intrinsics
- **reinterpret_cast (95):** Mostly in `__clang_cuda_device_functions.h` (CUDA builtins) and `c/parallel/src/` (JIT templates). These are largely unavoidable in GPU code.
- **c-style-pointer-cast (47):** Mix of CUDA device function wrappers and `c/parallel` code. The CUDA ones in vendor headers are not fixable; the `c/parallel/src/` ones could be modernized.
- **const_cast (33):** Review for cases where const-correctness could be improved upstream
- **Low unique rule hits:** Most semgrep security rules (buffer overflow, TOCTOU, etc.) didn't fire — indicates CCCL doesn't use many of the classic unsafe C patterns

## Reproduction

```bash
nix build .#analysis-semgrep-cpp
cat result/report.txt
```
