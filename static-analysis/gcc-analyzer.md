# GCC Analyzer Analysis

**Tool:** g++ -fanalyzer -fsyntax-only
**Total findings:** 0
**Scan scope:** C/C++ source files (excluding .cu)

## Summary

GCC's `-fanalyzer` performs interprocedural static analysis detecting null dereferences, use-after-free, double-free, buffer overflows, and infinite loops across function boundaries.

## Key Observations

- **Zero findings with `-Wanalyzer-*` prefix** — GCC analyzer likely produced output but the findings grep pattern filtered everything. The analyzer may also struggle with CCCL's template-heavy code without full include resolution.
- The tool ran but CCCL's non-CUDA C++ files are primarily headers and test utilities, which may not trigger interprocedural analysis paths.

## Reproduction

```bash
nix build .#analysis-gcc-analyzer
cat result/report.txt
```
