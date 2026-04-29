# PR Strategy for ShellCheck Fixes

## Overview

Four issues, each mapping to one PR. Ordered from most to least defensible. All changes are in `ci/` shell scripts only — no C++/CUDA code is modified.

## Issues → PRs

| # | Issue Title | Template | Files | Severity |
|---|-------------|----------|-------|----------|
| 1 | Unquoted array expansions in CI shell scripts | `[BUG]` | 5 | Error — genuine bugs |
| 2 | `if $(git rev-parse ...)` executes output as command | `[BUG]` | 1 | Warning — works by coincidence |
| 3 | Declaration commands mask return values (SC2155) | `[INFRA]` | 37 | Warning — undermines `set -e` |
| 4 | Shell script robustness improvements | `[INFRA]` | 7 | Note/Warning — defensive hardening |

## Recommended Filing Order

1. **Issues 1 + 2 together as one PR** — Small, focused, objectively correct. Best first impression.
2. **Issue 3 as a separate PR** — Large but mechanical. Easy to review once the pattern is understood.
3. **Issue 4 as a separate PR** — Mixed bag of small improvements. Least controversial.

## PR Template

Each PR should:
- Reference the filed issue with `closes #NNN`
- Include "Found via ShellCheck 0.11.0" in the description
- Note that all scripts use `set -euo pipefail` (establishes that strict error handling is intentional)
- Link to the ShellCheck wiki for each code (e.g., https://www.shellcheck.net/wiki/SC2068)

## Commit Plan

### PR 1: Fix shell script bugs (Issues 1 + 2)
```
Commit 1: "Fix unquoted array expansions in CI scripts (SC2068, SC2199, SC2145)"
  - ci/upload_cub_test_artifacts.sh
  - ci/test_cub.sh
  - ci/test_thrust.sh
  - ci/upload_thrust_test_artifacts.sh
  - ci/util/extract_switches.sh

Commit 2: "Fix accidental command execution in build_cuda_cccl_wheel.sh (SC2091)"
  - ci/build_cuda_cccl_wheel.sh
```

### PR 2: Fix masked return values (Issue 3)
```
Commit 1: "Separate readonly/local/export from command substitution in CI scripts (SC2155)"
  - All 37 affected files
```

### PR 3: Shell script robustness (Issue 4)
```
Commit 1: "Fix ambiguous quoting and minor shell issues in CI scripts"
  - ci/build_common.sh (SC2140 quoting fix)
  - ci/upload_cub_test_artifacts.sh (SC2207 mapfile)
  - ci/build_cub.sh (SC2206 quoted array elements)
  - ci/test_libcudacxx.sh (SC2164 cd error handling)
  - ci/util/build_and_test_targets.sh (SC2028 printf)
  - ci/build_cuda_cccl_python.sh (SC2046 quoted substitution)
```

## Intentionally Skipped

| Finding | Reason |
|---------|--------|
| SC2124 (build_common.sh) | Fixing requires changing how `CMAKE_OPTIONS` propagates through configure_preset/configure_and_build_preset. Too invasive for this PR. |
| SC2046 (pytorch/build_pytorch.sh) | Intentional word splitting — xargs output must be separate arguments to ninja |
| SC2148 (sourced files) | Missing shebang is expected for files meant to be `source`d, not executed |
| SC1091 (source not found) | Analysis artifact — files exist at runtime |
| SC2034 (unused variables) | Cross-script variables consumed by sourcing scripts |
| SC2154 (unassigned variables) | Variables come from sourced files |
| SC2086 (unquoted variables) | Many are intentional word splitting in CMake flag passing |

## Tone Guide

- Frame as "improving robustness" not "fixing bugs" (except for the genuine SC2068/SC2091 errors)
- Reference ShellCheck wiki URLs for authority
- Note that all scripts already opt into strict mode (`set -euo pipefail`), which establishes the intent for correctness
- Acknowledge that current code works in practice — the fixes prevent future regressions
- Do not characterize the codebase negatively
