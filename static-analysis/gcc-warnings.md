# GCC Extended Warnings Analysis

**Tool:** g++ with ~40 aggressive warning flags + `-fsyntax-only`
**Total findings:** 15,032
**Scan scope:** C/C++ source files (excluding .cu)

## Summary

Compiles individual C/C++ files with aggressive warning flags to catch issues the default build misses. Uses `-fsyntax-only` (no code generation) since CUDA is unavailable. Only non-CUDA files (.c, .cc, .cpp) are analyzed.

## Findings by Warning Flag

| Warning | Count | Description |
|---------|-------|-------------|
| -Wundef | 6,912 | Undefined macro used in `#if` |
| -Wtemplate-body | 544 | Template instantiation issues |
| -Wunused-variable | 38 | Unused variables |
| -Wunused-result | 21 | Ignoring return value |
| -Wlogical-not-parentheses | 21 | `!x < y` instead of `!(x < y)` |
| -Wunused-parameter | 19 | Unused function parameters |
| -Wfloat-equal | 19 | Direct float comparison with `==` |
| -Wdeprecated-declarations | 16 | Use of deprecated API |
| -Wunknown-pragmas | 13 | Unrecognized `#pragma` directives |
| -Wdouble-promotion | 10 | Implicit float-to-double promotion |
| -Wformat | 9 | Printf format string issues |
| -Wpedantic | 7 | Strict standard compliance |
| -Wconversion | 5 | Implicit type conversions |
| -Wnarrowing | 2 | Narrowing conversions |
| -Wsign-conversion | 1 | Sign conversion issues |

## Key Observations

- **-Wundef dominates (46%):** Undefined macros in `#if` preprocessor checks — common in CUDA codebases where macros like `__CUDACC__`, `_CCCL_COMPILER_NVCC` are conditionally defined
- **Actionable warnings:**
  - **-Wlogical-not-parentheses (21):** Potential logic bugs
  - **-Wfloat-equal (19):** Direct float comparison — may cause subtle bugs
  - **-Wformat (9):** Printf format mismatches — potential UB
  - **-Wconversion (5) + -Wnarrowing (2):** Type safety issues
- **Limited scope:** Only .c/.cc/.cpp files analyzed — .cu files skipped (need nvcc)

## Reproduction

```bash
nix build .#analysis-gcc-warnings
cat result/report.txt
```
