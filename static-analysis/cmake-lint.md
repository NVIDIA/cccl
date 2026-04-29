# CMake Lint Analysis

**Tool:** gersemi (--check mode)
**Total findings:** 0
**Scan scope:** CMakeLists.txt and .cmake files

## Summary

Gersemi checks CMake files for formatting consistency. Zero findings indicates all CMake files comply with the configured formatting rules.

This is expected — CCCL uses gersemi as a pre-commit hook for CMake formatting enforcement.

## Reproduction

```bash
nix build .#analysis-cmake-lint
cat result/report.txt
```
