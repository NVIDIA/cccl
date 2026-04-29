# Yamllint Analysis

**Tool:** yamllint 1.37.x
**Total findings:** 527
**Scan scope:** All YAML files

## Summary

Yamllint checks YAML files for syntax validity and style consistency.

## Key Observations

- Findings are predominantly line-length and formatting issues in CI workflow YAML files (`.github/workflows/`)
- No structural/syntax errors detected
- The high count reflects the extensive GitHub Actions CI infrastructure in this repo

## Reproduction

```bash
nix build .#analysis-yamllint
cat result/report.txt
```
