# PR 2: Separate declaration from command substitution in CI scripts

## PR Title
`Separate declaration from command substitution in CI scripts (SC2155)`

## PR Body

```markdown
## Description

closes #NNN

All CI scripts use `set -euo pipefail` to fail on errors. However, 76 sites combine `readonly`, `local`, or `export` with command substitution, which masks the command's exit code:

```bash
# Before — readonly always returns 0, masking cd failure:
readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# After — command failure is visible to set -e:
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
```

The same applies to `local` and `export`:

```bash
# local:
local start_time
start_time=$(date +%s)

# export:
CXX="$(which g++)"
export CXX
```

This is a mechanical transformation: split the declaration from the assignment. Behavior is identical when the command succeeds. When the command fails, the script now correctly aborts instead of continuing with an empty/partial value.

### Scope

| Pattern | Count | Files |
|---------|-------|-------|
| `readonly var="$(cmd)"` | ~55 | ci/util/artifacts/*.sh, ci/util/workflow/*.sh, ci/upload_*.sh |
| `local var="$(cmd)"` | ~15 | ci/pretty_printing.sh, ci/util/memmon.sh, ci/pyenv_helper.sh |
| `export var="$(cmd)"` | ~3 | ci/build_cuda_cccl_wheel.sh |
| `readonly var=$(cat <<EOF)` | ~3 | ci/util/create_mock_job_env.sh, ci/util/version_compare.sh |

Most of the diff is the `readonly ci_dir` + `readonly usage` boilerplate that appears at the top of every utility script.

### Reference
- [SC2155](https://www.shellcheck.net/wiki/SC2155)

Found via ShellCheck 0.11.0.

## Checklist
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
```
