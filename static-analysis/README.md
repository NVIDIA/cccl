# CCCL Static Analysis Results

Static analysis of the CUDA C++ Core Libraries (CCCL) repository using 17 automated tools via Nix-based reproducible infrastructure.

## Overview

| Tool | Findings | Category | Status |
|------|----------|----------|--------|
| [flawfinder](flawfinder.md) | 2,765 | C/C++ Security | done |
| [cpplint](cpplint.md) | 9,001 | C/C++ Style | done |
| [cppcheck](cppcheck.md) | 209 | C/C++ Deep Analysis | done |
| [ruff](ruff.md) | 9,173 | Python Linting | done |
| [shellcheck](shellcheck.md) | 255 | Shell Script | done |
| [yamllint](yamllint.md) | 527 | YAML Validation | done |
| [cmake-lint](cmake-lint.md) | 0 | CMake Formatting | done |
| [clang-tidy](clang-tidy.md) | 88,838 | C/C++ Modernization | done (noisy — see report) |
| [clang-analyzer](clang-analyzer.md) | 0 | C/C++ Path Analysis | done (infra issue) |
| [semgrep-cpp](semgrep-cpp.md) | 928 | C/C++ Patterns | done |
| [semgrep-python](semgrep-python.md) | 52 | Python Security | done |
| [bandit](bandit.md) | 690 | Python Security | done |
| [pylint](pylint.md) | 4,212 | Python Quality | done |
| [gcc-warnings](gcc-warnings.md) | 15,032 | C/C++ Warnings | done |
| [gcc-analyzer](gcc-analyzer.md) | 0 | C/C++ Interprocedural | done |
| [iwyu](iwyu.md) | — | Include Hygiene | running |
| [coccinelle](coccinelle.md) | 0 | Semantic Patches | done |

## Priority Findings

### Security-Relevant
- **shellcheck:** 8 errors — unquoted array expansions in CI scripts (SC2068, SC2199)
- **cppcheck:** 14 errors — including uninitialized variables
- **flawfinder:** `getenv` (65), `system` (305) usage in production code
- **bandit:** 25 subprocess-related findings in CI/build scripts (B602, B603, B607)
- **semgrep-python:** 24 `open()` calls with potentially unsanitized paths

### Code Quality
- **cppcheck:** 18 uninitialized member variables, 72 missing `explicit` constructors
- **cpplint:** 8 thread-unsafe function calls
- **pylint:** 203 `no-member` findings (accessing non-existent attributes)
- **gcc-warnings:** 21 `-Wlogical-not-parentheses` (potential logic bugs), 19 `-Wfloat-equal`
- **semgrep-cpp:** 47 C-style pointer casts in non-vendor code
- **ruff:** 314 print statements left in Python code

### False Positive Heavy (Low Priority)
- **clang-tidy:** ~95,000 of 88,838 findings are CUDA diagnostic errors (missing toolkit in sandbox)
- **gcc-warnings:** 6,912 `-Wundef` from conditional CUDA macros
- **flawfinder:** `random`, `access`, `equal` are mostly STL algorithm names (not security functions)
- **ruff:** 4,000+ missing type annotations — project doesn't enforce strict typing
- **bandit:** 609 `assert_used` — appropriate in test code
- **cpplint:** Style differences with Google guide (CCCL has its own style)

## Running Analysis

```bash
# Enter dev shell (all tools available)
nix develop

# Quick tier (~5-15 min): flawfinder, cpplint, ruff, shellcheck, yamllint, cmake-lint
nix build .#analysis-quick

# Standard tier (~30-60 min): quick + clang-tidy, cppcheck, bandit, pylint
nix build .#analysis-standard

# Deep tier (~1-3 hours): all 17 tools
nix build .#analysis-deep

# Individual tools
nix build .#analysis-flawfinder
nix build .#analysis-cppcheck
nix build .#analysis-clang-tidy
# ... etc

# Results appear in ./result/ (symlink to nix store)
ls result/
```

## Compilation Database

The analysis infrastructure generates a synthetic `compile_commands.json` with 4,836 entries for tools that require compilation context (clang-tidy, cppcheck, iwyu, clang-analyzer). This is generated automatically when the CUDA toolkit is not available in the nix sandbox.

```bash
nix build .#analysis-compile-db
cat result/method.txt    # "cmake" or "synthetic"
```
