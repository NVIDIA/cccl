# ShellCheck Analysis

**Tool:** shellcheck 0.11.0
**Total findings:** 255 (97 fixed — see below)
**Scan scope:** Shell scripts (bash dialect)

## Summary

ShellCheck identifies bugs, pitfalls, and style issues in shell scripts. All findings are in CI scripts under `ci/`.

## Fixes Applied

97 findings were fixed in commit `bf5958863` ([PR #8739](https://github.com/NVIDIA/cccl/pull/8739)):

| Fix | Findings | ShellCheck Codes |
|-----|----------|-----------------|
| Quote array expansions in `for` loops | 4 | SC2068 |
| Use `[*]` for explicit concatenation in `[[ ]]` | 3 | SC2199 |
| Use `[*]` in string context | 1 | SC2145 |
| Fix accidental command execution | 1 | SC2091 |
| Separate declaration from command substitution | 76 | SC2155 |
| Fix ambiguous quoting | 2 | SC2140 |
| Use `mapfile` for array deduplication | 1 | SC2207 |
| Quote variables in array assignment | 4 | SC2206 |
| Add `cd` error handling | 1 | SC2164 |
| Use `printf` for escape sequences | 1 | SC2028 |
| Quote command substitution | 1 | SC2046 |
| Fix companion quoting issues | 2 | SC1078, SC1079 |

**Remaining after fix:** 158 findings (69 SC1091 sourced-file-not-found, 59 SC2086 intentional word splitting, 12 SC2034 cross-script variables, 6 SC2154 sourced variables, and other intentional patterns).

## Findings by Severity

| Severity | Count |
|----------|-------|
| error | 8 |
| warning | 113 |
| note | 134 |

## All 21 Issues Explained

### 1. SC2155 — Masked return values (81 warnings)

**What it means:** When you combine `local` or `readonly` with command substitution on the same line, the exit code of the command is masked by the exit code of `local`/`readonly` (which is always 0). If the command fails, your script won't notice.

**Example from `ci/upload_cub_test_artifacts.sh:10`:**
```bash
# Bad — if `cd` or `pwd` fails, $? is still 0:
readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fixed — failure is now detectable:
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
```

**Affected files:** `ci/pyenv_helper.sh`, `ci/upload_cub_test_artifacts.sh`, `ci/util/artifacts/download_packed.sh`, `ci/util/artifacts/upload.sh`, `ci/util/artifacts/upload/*.sh` (most artifact scripts)

**Risk:** Under `set -e`, a failing command inside `readonly x=$(cmd)` won't trigger script abort. This matters in CI where you want hard failures on unexpected errors.

---

### 2. SC1091 — Sourced file not found (69 notes)

**What it means:** ShellCheck can't follow `source ./some_file.sh` because the file wasn't provided as input during analysis. Without seeing the sourced file, shellcheck can't track variables defined there.

**Example from `ci/build_thrust.sh:5`:**
```bash
source ./build_common.sh  # shellcheck can't see this file
```

**Affected files:** Nearly all `ci/*.sh` scripts source `./build_common.sh`, `./pretty_printing.sh`, or `./util/artifacts/common.sh`

**Risk:** None — this is a false positive from running shellcheck on files individually. The sourced files do exist at runtime. Can be suppressed with `# shellcheck source=./build_common.sh` directives.

---

### 3. SC2086 — Unquoted variable expansion (59 notes)

**What it means:** Unquoted variables undergo word splitting and glob expansion. If a variable contains spaces or glob characters, the shell will split it into multiple arguments or expand wildcards unexpectedly.

**Example from `ci/build_common.sh:170`:**
```bash
# Bad — if $CMAKE_OPTIONS contains spaces, it splits:
cmake $CMAKE_OPTIONS ...

# Fixed:
cmake "$CMAKE_OPTIONS" ...
# Or if intentional word splitting is needed:
cmake ${CMAKE_OPTIONS[@]} ...
```

**Affected files:** `ci/build_common.sh` (most occurrences), `ci/build_cub.sh`, `ci/build_cudax.sh`, `ci/test_cub.sh`, `ci/test_cuda_coop_python.sh`

**Risk:** Medium in CI — paths with spaces in Docker images, unexpected glob expansion in build directories. Usually works because CI paths are controlled, but fragile.

---

### 4. SC2034 — Apparently unused variable (12 warnings)

**What it means:** A variable is assigned but never referenced in the same script. Often these are used in scripts that source this file, or are environment variables consumed by child processes.

**Example from `ci/build_common.sh:82`:**
```bash
DISABLE_CUB_BENCHMARKS=ON  # Used by sourcing scripts
```

Other examples:
- `ci/build_common.sh:337` — `green` (color variable, used after sourcing)
- `ci/update_version.sh:22` — `pymajor` (consumed by other tools)
- `ci/build_cuda_cccl_python.sh:48` — `cuda12_image` (used in later sourced logic)

**Risk:** None in most cases — these are cross-script variables. Genuine dead code in a few cases (worth auditing `ci/verify_codegen_libcudacxx.sh:27 status` and `ci/matx/build_matx.sh:48 cccl_rapids_cmake_version`).

---

### 5. SC2154 — Variable referenced but not assigned (6 warnings)

**What it means:** A variable is used but never assigned in the current script. It's expected to come from a sourced file that shellcheck can't see.

**Example from `ci/test_cuda_coop_python.sh:14`:**
```bash
python${py_version} -m pytest ...  # py_version comes from sourced common_arg_parser.sh
```

**Affected files:** `ci/test_cuda_coop_python.sh`, `ci/build_cuda_cccl_python.sh`, `ci/build_cuda_cccl_wheel.sh`, `ci/test_cuda_cccl_examples_python.sh`, `ci/test_cuda_compute_python.sh`, `ci/test_cuda_cccl_headers_python.sh`

**Risk:** None — all instances are `py_version`, which is sourced from `./util/python/common_arg_parser.sh`. Related to SC1091.

---

### 6. SC2068 — Unquoted array expansion (4 errors)

**What it means:** Using `${array[@]}` without double quotes re-splits array elements. If an element contains spaces, it becomes multiple arguments.

**Example from `ci/upload_cub_test_artifacts.sh:53`:**
```bash
# Bad — elements with spaces get re-split:
for preset_variant in ${preset_variants[@]}; do

# Fixed:
for preset_variant in "${preset_variants[@]}"; do
```

**Affected files:** `ci/upload_cub_test_artifacts.sh:53`, `ci/test_cub.sh:136`, `ci/test_thrust.sh:59`, `ci/upload_thrust_test_artifacts.sh:36`

**Risk:** High — this is a genuine bug. If any preset variant name ever contains a space or glob character, the loop will malfunction. The fix is trivially adding double quotes.

---

### 7. SC2206 — Array assignment from unquoted command (4 warnings)

**What it means:** Assigning `array=($(command))` without quoting relies on word splitting and is vulnerable to glob expansion. Files named `*` in the current directory could be matched.

**Example from `ci/build_cub.sh:90-93`:**
```bash
# Bad:
CUB_BENCH_FLAGS=($CUB_BENCH_CUB_FLAGS)

# Fixed:
read -ra CUB_BENCH_FLAGS <<< "$CUB_BENCH_CUB_FLAGS"
# Or with mapfile:
mapfile -t CUB_BENCH_FLAGS < <(echo "$CUB_BENCH_CUB_FLAGS")
```

**Affected files:** `ci/build_cub.sh` (lines 90-93, four consecutive array assignments)

**Risk:** Medium — glob expansion risk in build directories that may contain files with special characters.

---

### 8. SC2199 — Array concatenation in [[ ]] (3 errors)

**What it means:** Using `${array[@]}` inside `[[ ]]` implicitly concatenates all elements into a single string, then performs the test on that string. This is almost never what you want.

**Example from `ci/upload_cub_test_artifacts.sh:78`:**
```bash
# Bad — concatenates all elements then checks for " no_lid ":
if [[ " ${preset_variants[@]} " =~ " no_lid " ]]; then

# Fixed — use printf and grep, or a loop:
if printf '%s\n' "${preset_variants[@]}" | grep -qx "no_lid"; then
```

**Affected files:** `ci/upload_cub_test_artifacts.sh:78`, `ci/upload_thrust_test_artifacts.sh:47,71`

**Risk:** High — this is a subtle logic bug. The regex match happens against a concatenated string, which can accidentally match substrings across element boundaries. In practice, the current code likely works because the preset names are simple, but it's brittle.

---

### 9. SC2124 — Array assigned to string variable (2 warnings)

**What it means:** Assigning `var="${array[@]}"` or `var=${array[@]}` collapses the array into a single space-separated string, losing element boundaries.

**Example from `ci/build_common.sh:304`:**
```bash
# This loses array element boundaries:
local cmake_flags=${CMAKE_FLAGS[@]}
```

**Affected files:** `ci/build_common.sh:304,432`

**Risk:** Low-medium — if any cmake flag contains spaces (e.g., `-DVAR="value with spaces"`), the flag will be split incorrectly when the string is later expanded.

---

### 10. SC2140 — Ambiguous quoting (2 warnings)

**What it means:** A word of the form `"A"B"C"` has ambiguous quoting. Bash interprets this as `A` (quoted) + `B` (unquoted) + `C` (quoted), which is usually not the intent.

**Example from `ci/build_common.sh:381`:**
```bash
# Ambiguous — B part is unquoted:
cmake_flag="-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES""

# Fixed:
cmake_flag="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
```

**Affected files:** `ci/build_common.sh:381,383`

**Risk:** Low — works in practice because the variable values don't contain spaces, but the quoting is technically incorrect and could break with unusual architecture strings.

---

### 11. SC2185 — `find` without explicit path (2 notes)

**What it means:** Some implementations of `find` don't default to the current directory. Explicitly passing `.` ensures portable behavior.

**Example from `ci/util/artifacts/upload.sh:13`:**
```bash
# Bad — not all `find` implementations default to `.`:
find -regex "..."

# Fixed:
find . -regex "..."
```

**Affected files:** `ci/util/artifacts/upload.sh:13`, `ci/util/artifacts/upload_packed.sh:15`

**Risk:** Low — GNU find (used in CI) defaults to `.`, but POSIX portability is broken.

---

### 12. SC2046 — Unquoted command substitution (2 warnings)

**What it means:** The output of `$(command)` undergoes word splitting and glob expansion when unquoted. Similar to SC2086 but for command substitutions specifically.

**Example from `ci/build_cuda_cccl_python.sh:14`:**
```bash
# Bad — output is subject to splitting:
echo "Docker socket: " $(ls /var/run/docker.sock)

# Fixed:
echo "Docker socket: $(ls /var/run/docker.sock)"
```

**Affected files:** `ci/build_cuda_cccl_python.sh:14`, `ci/pytorch/build_pytorch.sh:123`

**Risk:** Medium — the pytorch case `ninja -C ./build $(xargs -a build/cuda_targets.txt)` is intentional word splitting (targets as separate args), but could misbehave if target names contain spaces.

---

### 13. SC2317 — Unreachable command (1 note)

**What it means:** A command appears to be unreachable based on control flow analysis. It may be dead code, or it may be invoked indirectly (e.g., via a function name stored in a variable).

**Example from `ci/build_common.sh:91`:**
```bash
exit 1  # appears unreachable after usage function
```

**Affected files:** `ci/build_common.sh:91`

**Risk:** None — likely a false positive from shellcheck not following the `usage` function's control flow (the function may call `exit` itself).

---

### 14. SC2207 — Array from command substitution without mapfile (1 warning)

**What it means:** `array=($(command))` should use `mapfile` or `read -a` instead, to avoid word splitting and glob expansion issues.

**Example from `ci/upload_cub_test_artifacts.sh:45`:**
```bash
# Bad — glob expansion risk:
preset_variants=($(echo "${preset_variants[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

# Fixed:
mapfile -t preset_variants < <(printf '%s\n' "${preset_variants[@]}" | sort -u)
```

**Affected files:** `ci/upload_cub_test_artifacts.sh:45`

**Risk:** Medium — this deduplication line could misbehave if variant names match filenames as globs. The `mapfile` fix is cleaner and safer.

---

### 15. SC2164 — `cd` without error handling (1 warning)

**What it means:** If `cd` fails (directory doesn't exist, permissions issue), the script continues in the wrong directory, silently running subsequent commands in an unexpected location.

**Example from `ci/test_libcudacxx.sh:3`:**
```bash
# Bad — if cd fails, the rest of the script runs in the wrong dir:
cd "$(dirname "${BASH_SOURCE[0]}")"

# Fixed:
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
```

**Affected files:** `ci/test_libcudacxx.sh:3`

**Risk:** Medium — under `set -e` (which this script uses), the failure would abort. But if `set -e` is ever removed or disabled in a subshell, this becomes dangerous. The `|| exit 1` makes the intent explicit.

---

### 16. SC2145 — Mixing string and array in arguments (1 error)

**What it means:** When `"${array[@]}"` is used inside a string concatenation, the first element is joined with the prefix, but subsequent elements become separate arguments. This is almost never intended.

**Example from `ci/util/extract_switches.sh:67`:**
```bash
# Bad — first element joins with prefix, rest are separate args:
echo "${found_switches[@]} -- ${other_args[@]}"

# Fixed — use * to concatenate all into one string:
echo "${found_switches[*]} -- ${other_args[*]}"
```

**Affected files:** `ci/util/extract_switches.sh:67`

**Risk:** High — this is a genuine bug. The `echo` receives a fragmented argument list instead of a single formatted string. Elements after the first in each array become separate arguments to `echo`, which happens to still work for `echo` (it joins with spaces), but would break with other commands.

---

### 17. SC2102 — Range can only match single chars (1 note)

**What it means:** In `[[ ]]` or `case` patterns, character ranges like `[a-z]` can only match single characters, not multi-character strings.

**Example from `ci/test_python_common.sh:19`:**
```bash
pytest -n ${PARALLEL_LEVEL} -v ./tests
```

**Affected files:** `ci/test_python_common.sh:19`

**Risk:** None — likely a false positive triggered by the surrounding context rather than an actual range issue.

---

### 18. SC2091 — Executing command output (1 warning)

**What it means:** `$(command)` on its own line executes the output of `command` as a new command. This is usually a mistake — the intent was likely to just run `command` and check its exit code.

**Example from `ci/build_cuda_cccl_wheel.sh:26`:**
```bash
# Bad — executes the output of git rev-parse as a command:
if $(git rev-parse --is-shallow-repository); then

# Fixed — use the exit code directly:
if git rev-parse --is-shallow-repository; then
```

**Affected files:** `ci/build_cuda_cccl_wheel.sh:26`

**Risk:** High — `git rev-parse --is-shallow-repository` outputs "true" or "false" as text. The `$()` then executes `true` or `false` as a command, which happens to work by coincidence (both are valid shell builtins). But it's fragile — any unexpected output would be executed as a command.

---

### 19. SC2028 — `echo` may not expand escape sequences (1 note)

**What it means:** `echo` behavior with escape sequences (`\n`, `\t`) is not portable. Some shells expand them, others don't. Use `printf` for reliable behavior.

**Example from `ci/util/build_and_test_targets.sh:89`:**
```bash
# Bad — \n and \t may not be expanded:
echo "🔴📝 Configuration override failed ($(elapsed_time)):\n\t${CONFIGURE_OVERRIDE}"

# Fixed:
printf '🔴📝 Configuration override failed (%s):\n\t%s\n' "$(elapsed_time)" "${CONFIGURE_OVERRIDE}"
```

**Affected files:** `ci/util/build_and_test_targets.sh:89`

**Risk:** Low — the error message will show literal `\n\t` instead of a newline and tab in some environments.

---

### 20. SC1079 — Suspicious end quote (1 note)

**What it means:** An end quote looks like it might be misplaced due to the character immediately following it, creating ambiguous quoting.

**Example from `ci/build_common.sh:383`:**
```bash
# The quote structure is confusing:
cmake_flag="-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES""
#           ^--- open    close ---^   ^--- this looks like another open
```

**Affected files:** `ci/build_common.sh:383`

**Risk:** None — companion to SC2140 (issue #10). Same fix applies.

---

### 21. SC1078 — Unclosed double quote (1 warning)

**What it means:** ShellCheck thinks a double-quoted string was never closed. This is the flip side of the SC2140/SC1079 ambiguous quoting pattern.

**Example from `ci/build_common.sh:381`:**
```bash
# Shellcheck sees the first " and can't find its matching close:
cmake_flag="-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES""
```

**Affected files:** `ci/build_common.sh:381`

**Risk:** None — same root cause as SC2140 and SC1079 above. All three will be fixed together.

## Complete Issue Summary

| Code | Count | Severity | Verdict |
|------|-------|----------|---------|
| SC2155 | 81 | warning | Fix — masked return values under `set -e` |
| SC1091 | 69 | note | Ignore — expected without cross-file analysis |
| SC2086 | 59 | note | Review — most are intentional word splitting |
| SC2034 | 12 | warning | Audit — some are cross-script, some may be dead code |
| SC2154 | 6 | warning | Ignore — variables come from sourced files |
| SC2206 | 4 | warning | Fix — use `mapfile` or `read -a` |
| SC2068 | 4 | error | **Fix** — unquoted array expansion, real bug |
| SC2199 | 3 | error | **Fix** — array concatenation logic bug |
| SC2185 | 2 | note | Fix — add explicit `.` path to `find` |
| SC2140 | 2 | warning | Fix — ambiguous quoting |
| SC2124 | 2 | warning | Fix — array-to-string collapse |
| SC2046 | 2 | warning | Review — one intentional, one should be quoted |
| SC2317 | 1 | note | Ignore — false positive (indirect invocation) |
| SC2207 | 1 | warning | Fix — use `mapfile` for dedup |
| SC2164 | 1 | warning | Fix — add `\|\| exit 1` after `cd` |
| SC2145 | 1 | error | **Fix** — use `[*]` instead of `[@]` in string |
| SC2102 | 1 | note | Ignore — false positive |
| SC2091 | 1 | warning | **Fix** — executing command output by accident |
| SC2028 | 1 | note | Fix �� use `printf` instead of `echo` |
| SC1079 | 1 | note | Fix — companion to SC2140 |
| SC1078 | 1 | warning | Fix — companion to SC2140 |

## Reproduction

```bash
nix build .#analysis-shellcheck
cat result/report.txt
```
