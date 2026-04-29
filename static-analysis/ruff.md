# Ruff Analysis

**Tool:** ruff 0.15.x
**Total findings:** 9,173
**Scan scope:** All Python files (--select=ALL)

## Summary

Ruff is a fast Python linter that implements rules from dozens of linting tools (pyflakes, pycodestyle, pydocstyle, flake8 plugins, pylint, etc.). Running with `--select=ALL` enables all available rules for maximum coverage.

## Top 15 Rules

| Rule | Count | Description |
|------|-------|-------------|
| ANN001 | 2,238 | Missing type annotation for function argument |
| ANN201 | 1,016 | Missing return type annotation for public function |
| S101 | 609 | Use of `assert` (banned in security contexts) |
| ANN202 | 549 | Missing return type annotation for private function |
| TID252 | 327 | Relative imports from parent packages |
| T201 | 314 | `print()` found (use logging) |
| D212 | 268 | Multi-line docstring summary placement |
| ANN204 | 208 | Missing return type annotation for `__init__` |
| TRY003 | 146 | Long exception message outside exception class |
| PLR2004 | 144 | Magic value in comparison |
| D413 | 126 | Missing blank line after last section in docstring |
| UP031 | 125 | Use format specifiers instead of percent format |
| UP006 | 107 | Use `type` instead of `Type` from typing |
| D205 | 92 | Blank line between summary and description |
| INP001 | 85 | Implicit namespace package (missing `__init__.py`) |

## Key Observations

- **Type annotations dominate:** ANN001 + ANN201 + ANN202 + ANN204 = 4,011 (44% of all findings) — expected for a project not enforcing strict typing
- **Documentation style:** D-prefixed rules (pydocstyle) account for ~500 findings
- **Security noise:** S101 (`assert`) is mostly test code where `assert` is appropriate
- **Actionable:** T201 (print statements, 314), PLR2004 (magic numbers, 144), UP031/UP006 (modernization)
- **False positives:** Many `INP001` findings are in `.gersemi/ext/` and script directories that intentionally lack `__init__.py`

## Reproduction

```bash
nix build .#analysis-ruff
cat result/report.txt
```
