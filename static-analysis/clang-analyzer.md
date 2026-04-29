# Clang Static Analyzer

**Tool:** clang --analyze (via scan-build)
**Total findings:** 0
**Compile database:** Synthetic (4,836 translation units)

## Summary

The Clang Static Analyzer performs deep path-sensitive analysis to detect null dereferences, use-after-free, memory leaks, and other interprocedural bugs.

## Key Observations

- **Zero findings** — the analyzer failed to run because `clang` was not found in the nix build sandbox. The tool needs adjustment to reference the correct clang binary from the `clang-tools` package.
- This is an infrastructure issue, not a code quality result.

## Reproduction

```bash
nix build .#analysis-clang-analyzer
cat result/report.txt
```
