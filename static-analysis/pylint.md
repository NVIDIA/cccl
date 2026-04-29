# Pylint Analysis

**Tool:** pylint
**Total findings:** 4,212
**Scan scope:** All Python files

## Summary

Pylint performs comprehensive Python code quality analysis including style, error detection, refactoring suggestions, and convention enforcement.

## Top 20 Findings

| Message ID | Count | Description |
|------------|-------|-------------|
| missing-function-docstring | 906 | Missing docstring in function/method |
| invalid-name | 298 | Name doesn't conform to naming style |
| consider-using-f-string | 212 | Old-style string formatting |
| no-member | 203 | Accessing non-existent member |
| duplicate-code | 191 | Duplicated code blocks |
| line-too-long | 184 | Line exceeds max length |
| missing-module-docstring | 148 | Missing module docstring |
| wrong-import-position | 133 | Import not at top of file |
| missing-class-docstring | 124 | Missing class docstring |
| too-many-locals | 105 | Too many local variables |
| redefined-outer-name | 100 | Redefining name from outer scope |
| too-few-public-methods | 92 | Class with too few public methods |
| too-many-arguments | 83 | Too many function arguments |
| too-many-positional-arguments | 82 | Too many positional arguments |
| import-outside-toplevel | 80 | Import outside top level |
| protected-access | 75 | Accessing protected member |
| redefined-builtin | 64 | Redefining built-in name |
| fixme | 57 | TODO/FIXME comments |
| unused-argument | 55 | Unused function argument |
| no-else-return | 43 | Unnecessary `else` after `return` |

## Key Observations

- **Documentation (1,178):** missing-function/module/class-docstring — expected for utility scripts
- **Potential bugs (203):** `no-member` findings are the most actionable — accessing attributes that may not exist
- **Code complexity (280):** too-many-locals, too-many-arguments — refactoring candidates
- **Modernization (212):** `consider-using-f-string` — straightforward cleanup
- **Primary areas:** `ci/` scripts, `.github/actions/`, `libcudacxx/test/utils/`, benchmark code

## Reproduction

```bash
nix build .#analysis-pylint
cat result/report.txt
```
