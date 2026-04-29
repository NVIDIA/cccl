# Coccinelle Analysis

**Tool:** coccinelle (spatch)
**Total findings:** 0
**Scan scope:** C/C++ source files
**Rules:** null-deref, use-after-free, double-free, resource-leak

## Summary

Coccinelle uses semantic patches (.cocci files) to detect specific bug patterns: malloc without NULL check, free followed by pointer use, double-free, and fopen without fclose.

## Key Observations

- **Zero findings** — CCCL's C/C++ code does not use raw `malloc`/`free`/`fopen` patterns that the coccinelle rules target. The project uses C++ RAII patterns and smart pointers.
- This is a good result — indicates modern memory management practices.

## Reproduction

```bash
nix build .#analysis-coccinelle
cat result/report.txt
```
