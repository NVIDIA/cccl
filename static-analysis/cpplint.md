# Cpplint Analysis

**Tool:** cpplint (Google C++ Style Guide)
**Total findings:** 9,001 (1,702 unique errors after deduplication)
**Scan scope:** C/C++/CUDA source files

## Summary

Cpplint checks C++ source files against the Google C++ Style Guide. Many findings reflect style differences between Google's conventions and CCCL's established coding style.

## Findings by Category

| Category | Count | Description |
|----------|-------|-------------|
| build/include_what_you_use | 564 | Missing includes for used symbols |
| readability/braces | 224 | Brace placement issues |
| build/namespaces | 211 | Using-directives in namespaces |
| runtime/int | 211 | Use of `short`, `long` instead of fixed-width |
| runtime/explicit | 169 | Missing `explicit` on constructors |
| readability/casting | 107 | C-style casts |
| build/include_order | 106 | Include ordering violations |
| readability/check | 59 | Readability issues |
| readability/inheritance | 18 | Virtual function issues |
| runtime/threadsafe_fn | 8 | Thread-unsafe function usage |
| runtime/arrays | 6 | Array usage issues |
| runtime/printf | 3 | Printf format issues |
| runtime/string | 3 | String handling issues |
| readability/nolint | 2 | Orphaned NOLINT comments |
| build/c++17 | 2 | C++17 feature usage |

## Priority Issues

1. **runtime/threadsafe_fn (8):** Thread-unsafe function calls — review for concurrent usage
2. **readability/casting (107):** C-style casts should be `static_cast`/`reinterpret_cast`
3. **runtime/explicit (169):** Implicit conversions from single-arg constructors

## Key Observations

- **High noise for CCCL:** CCCL follows its own style guide, not Google's — many `build/namespaces`, `readability/braces`, and `runtime/int` findings are intentional style choices
- **Actionable subset:** `runtime/threadsafe_fn`, `readability/casting`, `build/include_what_you_use` are worth reviewing
- **Primary areas:** c/experimental/stf, c/parallel

## Reproduction

```bash
nix build .#analysis-cpplint
cat result/report.txt
```
