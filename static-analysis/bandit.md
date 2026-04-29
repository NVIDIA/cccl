# Bandit Analysis

**Tool:** bandit 1.9.x
**Total findings:** 690
**Scan scope:** All Python files

## Summary

Bandit is a Python security linter that finds common security issues. It maps findings to CWE identifiers.

## Findings by Issue

| Issue ID | Count | Description |
|----------|-------|-------------|
| B101:assert_used | 609 | Use of `assert` (disabled in optimized mode) |
| B603:subprocess_without_shell_equals_true | 22 | Subprocess call without shell=True check |
| B404:blacklist | 15 | Import of `subprocess` module |
| B607:start_process_with_partial_path | 10 | Process started with partial path |
| B110:try_except_pass | 7 | `try/except/pass` — silently swallowing exceptions |
| B314:blacklist | 6 | Use of xml.etree (potential XXE) |
| B102:exec_used | 6 | Use of `exec()` |
| B608:hardcoded_sql_expressions | 4 | Hardcoded SQL expressions |
| B602:subprocess_popen_with_shell_equals_true | 3 | Subprocess with `shell=True` |
| B307:blacklist | 3 | Use of `eval()` |
| B311:blacklist | 2 | Use of `random` (not cryptographically secure) |
| B405:blacklist | 1 | Import of xml.etree |
| B324:hashlib | 1 | Use of insecure hash (SHA-1/MD5) |
| B103:set_bad_file_permissions | 1 | Insecure file permissions |

## Key Observations

- **B101 dominates (88%):** `assert` usage is appropriate in test code — not a real security issue for this project
- **Actionable security findings (~30):**
  - B602/B603 (25): Subprocess calls — review for command injection in CI scripts
  - B110 (7): Silent exception swallowing — may hide errors
  - B608 (4): Hardcoded SQL — likely string formatting, not actual SQL injection
  - B324 (1): Insecure hash — check if used for security purposes
- **Primary areas:** `.devcontainer/launch.py`, `ci/` scripts, `.github/actions/` workflows

## Reproduction

```bash
nix build .#analysis-bandit
cat result/report.txt
```
