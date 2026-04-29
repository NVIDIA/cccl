# Semgrep Python Analysis

**Tool:** semgrep with vendored Python security rules
**Total findings:** 52
**Scan scope:** All Python files

## Summary

Semgrep uses 20 pattern-based rules to detect security issues in Python code: command injection, path traversal, insecure deserialization, hardcoded secrets, and more.

## Findings by Rule

| Rule | Count | Description |
|------|-------|-------------|
| py-print-statement | 27 | `print()` in production code (prefer logging) |
| py-open-user-input | 24 | `open()` with potentially unsanitized path |
| py-sha1-usage | 1 | Use of SHA-1 (insecure for cryptographic purposes) |

## Key Observations

- **py-open-user-input (24):** File opens where the path comes from function arguments or variables — review whether paths are user-controlled. Most are in CI/build scripts where paths come from trusted sources.
- **py-print-statement (27):** Overlaps with ruff's T201 finding. In CI scripts, `print()` is often intentional for logging.
- **py-sha1-usage (1):** Check if SHA-1 is used for integrity/security or just as a non-security hash.
- **No high-severity findings:** No command injection, SQL injection, deserialization, or hardcoded secret patterns detected.

## Reproduction

```bash
nix build .#analysis-semgrep-python
cat result/report.txt
```
