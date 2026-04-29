# Cppcheck Analysis

**Tool:** cppcheck (with compilation database)
**Total findings:** 209
**Compile database:** Synthetic (4,836 translation units)

## Summary

Cppcheck performs deep static analysis of C/C++ code using a compilation database for accurate type resolution. This analysis used a synthetically-generated compile_commands.json since the full CUDA build was not available.

## Findings by Severity

| Severity | Count |
|----------|-------|
| style | 147 |
| performance | 26 |
| warning | 21 |
| information | 14 |
| error | 14 |
| portability | 1 |

## Top Checks

| Check ID | Count | Description |
|----------|-------|-------------|
| noExplicitConstructor | 72 | Single-argument constructors should be `explicit` |
| functionConst | 55 | Member function can be const-qualified |
| uninitMemberVarPrivate | 18 | Uninitialized member variable in constructor |
| functionStatic | 16 | Member function can be static |
| normalCheckLevelMaxBranches | 14 | Branch analysis limit reached |
| passedByValue | 10 | Argument should be passed by const reference |
| syntaxError | 8 | Parse errors (likely CUDA/template syntax) |
| constVariableReference | 8 | Variable reference can be const |
| unknownMacro | 4 | Unrecognized macros |
| useStlAlgorithm | 3 | Loop can be replaced with STL algorithm |

## Priority Issues

1. **uninitMemberVarPrivate (18):** Uninitialized members in `BuildInformation` and other test structures — potential undefined behavior
2. **error severity (14):** Includes syntax errors from CUDA-specific syntax that cppcheck can't parse, plus real issues
3. **passedByValue (10):** Performance issue — large objects copied unnecessarily

## Key Observations

- The synthetic compile database enables analysis but may miss some include paths, causing `syntaxError` and `unknownMacro` findings
- Most findings are in `c/parallel/test/` code
- The `noExplicitConstructor` findings (72) are largely style — review for actual implicit conversion risks

## Reproduction

```bash
nix build .#analysis-cppcheck
cat result/report.txt
```
