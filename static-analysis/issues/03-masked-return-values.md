# Issue: Masked return values in CI shell scripts (SC2155)

**Template:** `[INFRA]: Declaration commands mask return values in CI scripts`
**Component:** Infrastructure
**Severity:** Low (no known failures, but undermines set -e guarantees)

## Description

76 sites across 37 CI scripts combine variable declaration (`local`, `readonly`, or `export`) with command substitution on the same line. When these are combined, the exit code of the command substitution is masked by the exit code of the declaration keyword, which is always 0.

All affected scripts use `set -euo pipefail`, explicitly opting into strict error handling. Combining declaration with substitution silently defeats that intent — if the command fails, the script continues with an empty or partial value instead of aborting.

## Pattern

```bash
# Current — if `cd` fails, $? is still 0 due to `readonly`:
readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fixed — failure in command substitution is now visible to set -e:
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
```

The same applies to `local` and `export`:

```bash
# local masks return value:
local start_time=$(date +%s)
# Fixed:
local start_time
start_time=$(date +%s)

# export masks return value:
export CXX="$(which g++)"
# Fixed:
CXX="$(which g++)"
export CXX
```

## Scope

The 76 findings fall into three groups:

| Pattern | Count | Files | Notes |
|---------|-------|-------|-------|
| `readonly ci_dir="$(cd ...)"` | ~30 | ci/util/artifacts/*.sh, ci/util/workflow/*.sh, ci/upload_*.sh | Identical boilerplate in utility scripts |
| `readonly usage=$(cat <<EOF` | ~25 | Same files as ci_dir | Usage string assignment |
| `local`/`export` + substitution | ~21 | ci/pretty_printing.sh, ci/util/memmon.sh, ci/build_cuda_cccl_wheel.sh, ci/matx/build_matx.sh, ci/pyenv_helper.sh, ci/update_version.sh | Various unique patterns |

## Risk Assessment

The fix separates declaration from assignment, which is a mechanical transformation. The resulting behavior is identical when the command succeeds, and strictly better when the command fails (the script will abort instead of continuing with a bad value).

The `readonly usage=$(cat <<EOF ... EOF)` pattern is a special case: `cat <<EOF` cannot fail in practice (it reads from the script itself), so masking its return value is harmless. Fixing it anyway maintains consistency and avoids a "why is this one different?" question during review.

## Verification

Run `shellcheck -e SC1091 -i SC2155 ci/**/*.sh` before and after to confirm all SC2155 findings are resolved.

## Detection

Found via ShellCheck 0.11.0 (SC2155: "Declare and assign separately to avoid masking return values"). Full analysis: `static-analysis/shellcheck.md`.
