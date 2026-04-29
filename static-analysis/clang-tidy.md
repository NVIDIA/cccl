# Clang-Tidy Analysis

**Tool:** clang-tidy (with compilation database)
**Total findings:** 88,838
**Compile database:** Synthetic (4,836 translation units)

## Summary

Clang-tidy performs comprehensive linting and modernization checks using CCCL's existing `.clang-tidy` configuration. Results are heavily impacted by running without CUDA toolkit — most findings are diagnostic errors from missing CUDA headers/intrinsics.

## Findings by Check

| Check | Count | Description |
|-------|-------|-------------|
| clang-diagnostic-error | 95,037 | Compilation errors (missing CUDA, headers) |
| nodiscard | 2,305 | Missing `[[nodiscard]]` attribute usage |
| noreturn | 2,986 | Missing `[[noreturn]]` attribute |

## Key Observations

- **~95,000 are CUDA diagnostic errors:** The synthetic compile database lacks CUDA toolkit, so `.cu` files and CUDA-dependent `.cpp` files produce `cannot find CUDA installation`, `unknown type __global__`, etc. These are infrastructure noise, not real issues.
- **Actionable findings (~5,000):** The `nodiscard` and `noreturn` findings are legitimate modernization suggestions
- **CCCL's `.clang-tidy` config was used:** The analysis respected the project's check selection (bugprone, cert, clang-analyzer, concurrency, cppcoreguidelines, misc, modernize, performance, readability)
- **For accurate results:** Run with CUDA toolkit available, or filter out `clang-diagnostic-error` findings

## Filtering Noise

To see only actionable findings:
```bash
grep -v 'clang-diagnostic-error' result/report.txt | grep ': warning:\|: error:'
```

## Reproduction

```bash
nix build .#analysis-clang-tidy
cat result/report.txt
```
