# Flawfinder Analysis

**Tool:** flawfinder 2.0.19
**Total findings:** 2,765
**Scan scope:** C/C++/CUDA source files

## Summary

Flawfinder scans source code for potential security vulnerabilities using a built-in database of 222 C/C++ dangerous function patterns, mapped to CWE identifiers.

## Findings by Category

| Function/Pattern | Count | Primary CWE |
|-----------------|-------|-------------|
| read | 738 | CWE-120 (Buffer overflow) |
| random | 657 | CWE-327 (Broken crypto) |
| system | 305 | CWE-78 (OS command injection) |
| equal | 305 | CWE-126 |
| char | 155 | CWE-120 (Buffer overflow) |
| access | 123 | CWE-362 (Race condition) |
| atoi | 93 | CWE-190 (Integer overflow) |
| memcpy | 77 | CWE-120 (Buffer overflow) |
| is_permutation | 69 | CWE-126 |
| getenv | 65 | CWE-807 (Untrusted input) |
| strncpy | 38 | CWE-120 (Buffer overflow) |
| atol | 29 | CWE-190 (Integer overflow) |
| mismatch | 25 | CWE-126 |
| fopen | 19 | CWE-362 (Race condition) |
| strlen | 17 | CWE-126 |

## Key Observations

- **High false-positive rate expected:** Many `random`, `access`, `equal`, `is_permutation`, and `mismatch` hits are C++ STL algorithm names that happen to match flawfinder's pattern database (e.g., `std::equal`, `accessor().access()`)
- **Legitimate concerns:** `getenv` (65), `system` (305), `memcpy` (77) in non-test code warrant review
- **Test code dominates:** Many `random` findings are from Catch2 `GENERATE(take(N, random(...)))` test generators
- **Primary project areas:** c/parallel (hostjit, tests), libcudacxx (string tests)

## Reproduction

```bash
nix build .#analysis-flawfinder
cat result/report.csv
```
