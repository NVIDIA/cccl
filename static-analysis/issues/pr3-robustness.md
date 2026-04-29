# PR 3: Shell script robustness improvements in CI

## PR Title
`Shell script robustness improvements in CI`

## PR Body

```markdown
## Description

closes #NNN

Minor shell script improvements found via ShellCheck 0.11.0. Each fix is independent and low-risk.

### Changes

**Ambiguous quoting in `build_common.sh`** (SC2140): `"weighted"` and `"important"` inside a double-quoted echo string have incorrect quoting. The unquoted portions undergo word splitting. Fixed with escaped quotes.

**Array deduplication in `upload_cub_test_artifacts.sh`** (SC2207): Replaced `array=($(... | sort -u))` with `mapfile -t array < <(... | sort -u)`. The mapfile form avoids word splitting and glob expansion during array assignment.

**Quoted CMake flag variables in `build_cub.sh`** (SC2206): Added double quotes around variable expansions inside array assignment to prevent word splitting.

**`cd` error handling in `test_libcudacxx.sh`** (SC2164): Added `|| exit 1` after `cd` to make failure explicit. The script uses `set -e` which would catch this, but the explicit form survives if error handling is ever adjusted.

**`printf` instead of `echo` for escape sequences in `build_and_test_targets.sh`** (SC2028): `echo` does not reliably expand `\n` and `\t`. Replaced with `printf` for portable formatting.

**Quoted command substitution in `build_cuda_cccl_python.sh`** (SC2046): Moved `$(ls ...)` inside the double-quoted string to prevent word splitting.

### Intentionally skipped

- SC2124 in `build_common.sh` (`local CMAKE_OPTIONS=$@`): Fixing requires changing how cmake options propagate through `configure_preset` and `configure_and_build_preset`. Deferred as a separate effort.
- SC2046 in `pytorch/build_pytorch.sh`: The unquoted `$(xargs ...)` is intentional — each target must be a separate argument to ninja.

### References
- [SC2140](https://www.shellcheck.net/wiki/SC2140), [SC2207](https://www.shellcheck.net/wiki/SC2207), [SC2206](https://www.shellcheck.net/wiki/SC2206)
- [SC2164](https://www.shellcheck.net/wiki/SC2164), [SC2028](https://www.shellcheck.net/wiki/SC2028), [SC2046](https://www.shellcheck.net/wiki/SC2046)

Found via ShellCheck 0.11.0.

## Checklist
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
```
