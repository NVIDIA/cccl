# PR 1: Fix shell script array handling and command execution bugs in CI

## PR Title
`Fix shell script array handling and command execution bugs in CI`

## PR Body

```markdown
## Description

closes #NNN

Fix 9 ShellCheck error-level findings in CI shell scripts. These are genuine bugs that work by coincidence with current values but would break under edge cases.

### Array quoting (SC2068, SC2199, SC2145)

Several `for` loops iterate over arrays without quoting the expansion. Without double quotes, array elements are re-split on whitespace and subject to glob expansion:

```bash
# Before — elements re-split on spaces:
for preset_variant in ${preset_variants[@]}; do

# After — element boundaries preserved:
for preset_variant in "${preset_variants[@]}"; do
```

Current values (`no_lid`, `lid_0`, etc.) contain no spaces, so this works today. The quoted form is strictly safer and preserves identical behavior.

Three `[[ ]]` tests use `${array[@]}` which implicitly concatenates elements. Changed to `${array[*]}` to make the concatenation explicit (identical behavior, clearer intent).

One `echo` in `extract_switches.sh` uses `${array[@]}` in a string context. Changed to `${array[*]}` for consistency.

**Files:** `ci/upload_cub_test_artifacts.sh`, `ci/test_cub.sh`, `ci/test_thrust.sh`, `ci/upload_thrust_test_artifacts.sh`, `ci/util/extract_switches.sh`

### Command execution (SC2091)

`ci/build_cuda_cccl_wheel.sh` uses `if $(git rev-parse --is-shallow-repository)` which executes the *output* of the command ("true"/"false") as a shell command. This works because `true` and `false` are shell builtins, but it's accidental — any unexpected output would be executed:

```bash
# Before — executes output as command:
if $(git rev-parse --is-shallow-repository); then

# After — string comparison:
if [ "$(git rev-parse --is-shallow-repository)" = "true" ]; then
```

Note: `git rev-parse --is-shallow-repository` always exits 0 regardless of answer, so the exit code can't be used directly. String comparison is the correct approach.

### References
- [SC2068](https://www.shellcheck.net/wiki/SC2068)
- [SC2199](https://www.shellcheck.net/wiki/SC2199)
- [SC2145](https://www.shellcheck.net/wiki/SC2145)
- [SC2091](https://www.shellcheck.net/wiki/SC2091)

Found via ShellCheck 0.11.0.

## Checklist
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
```
