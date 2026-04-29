# Issue: Unquoted array expansions in CI shell scripts

**Template:** `[BUG]: Unquoted array expansions in CI shell scripts`
**Component:** Infrastructure
**Severity:** Low (no current failures, but fragile under edge cases)

## Description

Several CI scripts use `${array[@]}` without double quotes in `for` loops. Without quotes, array elements are subject to word splitting and glob expansion — if an element ever contains a space or glob character (`*`, `?`, `[`), the loop will silently malfunction.

The current values (`no_lid`, `lid_0`, `cub-nolid`, `thrust-cpu`, etc.) contain no spaces, so this works today. The fix is adding double quotes, which preserves current behavior while making the code robust against future changes to these values.

Additionally, three sites use `${array[@]}` inside `[[ ]]` for array membership checks. In this context, `[@]` implicitly concatenates elements into a single string before the regex match. The code works correctly because the concatenated string is space-separated and the regex pads with spaces, but using `[*]` makes the concatenation intent explicit.

## Affected Files

### Unquoted `for` loops (SC2068) — 4 sites

**`ci/upload_cub_test_artifacts.sh:53`**
```bash
# Current:
for preset_variant in ${preset_variants[@]}; do

# Proposed:
for preset_variant in "${preset_variants[@]}"; do
```

**`ci/test_cub.sh:136-137`**
```bash
# Current:
for PRESET in ${PRESETS[@]}; do
  test_preset "CUB (${PRESET})" ${PRESET}

# Proposed:
for PRESET in "${PRESETS[@]}"; do
  test_preset "CUB (${PRESET})" "${PRESET}"
```

**`ci/test_thrust.sh:59-60`**
```bash
# Current:
for PRESET in ${PRESETS[@]}; do
  test_preset "Thrust (${PRESET})" ${PRESET} ${GPU_REQUIRED}

# Proposed:
for PRESET in "${PRESETS[@]}"; do
  test_preset "Thrust (${PRESET})" "${PRESET}" "${GPU_REQUIRED}"
```

**`ci/upload_thrust_test_artifacts.sh:36`**
```bash
# Current:
for preset_variant in ${preset_variants[@]}; do

# Proposed:
for preset_variant in "${preset_variants[@]}"; do
```

### Implicit array concatenation in `[[ ]]` (SC2199) — 3 sites

These use `${array[@]}` in `[[ ]]` to check array membership. The `[@]` form implicitly concatenates elements; `[*]` makes concatenation explicit. Both produce identical results here.

**`ci/upload_cub_test_artifacts.sh:78`**
```bash
# Current:
if [[ " ${preset_variants[@]} " =~ " no_lid " ]]; then

# Proposed:
if [[ " ${preset_variants[*]} " =~ " no_lid " ]]; then
```

**`ci/upload_thrust_test_artifacts.sh:47,71`**
```bash
# Current:
if [[ " ${preset_variants[@]} " =~ " test_cpu " ]]; then
if [[ " ${preset_variants[@]} " =~ " test_gpu " ]]; then

# Proposed:
if [[ " ${preset_variants[*]} " =~ " test_cpu " ]]; then
if [[ " ${preset_variants[*]} " =~ " test_gpu " ]]; then
```

### Array in string context (SC2145) — 1 site

**`ci/util/extract_switches.sh:67`**
```bash
# Current:
echo "${found_switches[@]} -- ${other_args[@]}"

# Proposed:
echo "${found_switches[*]} -- ${other_args[*]}"
```

Within `echo`, `[@]` expands each element as a separate argument; `echo` re-joins them with spaces, producing identical output to `[*]`. The `[*]` form makes the "concatenate into a string" intent explicit.

## Verification

All fixes preserve current behavior. The `for` loop quoting changes are strictly safer (no word splitting of elements). The `[*]` vs `[@]` changes produce identical output in these contexts.

To verify: run `shellcheck -e SC1091 ci/upload_cub_test_artifacts.sh ci/test_cub.sh ci/test_thrust.sh ci/upload_thrust_test_artifacts.sh ci/util/extract_switches.sh` before and after.

## Detection

Found via ShellCheck 0.11.0 (SC2068, SC2199, SC2145). Full analysis: `static-analysis/shellcheck.md`.
