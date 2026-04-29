# Issue: Shell script robustness improvements in CI

**Template:** `[INFRA]: Shell script robustness improvements in CI scripts`
**Component:** Infrastructure
**Severity:** Low (cosmetic and defensive improvements)

## Description

Several minor shell script issues detected via ShellCheck that improve robustness and correctness. Each fix is independent and low-risk.

## Fixes

### 1. Array-to-string assignment in build_common.sh (SC2124) — 2 sites

`local CMAKE_OPTIONS=$@` assigns an array to a scalar variable, losing element boundaries. This happens to work because the variable is later expanded unquoted (re-splitting on spaces), but a cmake flag containing spaces (e.g., `-DVAR="a b"`) would break.

**`ci/build_common.sh:304`** (in `configure_preset`):
```bash
# Current:
local CMAKE_OPTIONS=$@

# Proposed:
local CMAKE_OPTIONS=("$@")
```

**`ci/build_common.sh:432`** (in `configure_and_build_preset`):
```bash
# Current:
local CMAKE_OPTIONS=$@

# Proposed:
local CMAKE_OPTIONS=("$@")
```

Note: Downstream usage of `$CMAKE_OPTIONS` must also be adjusted to use `"${CMAKE_OPTIONS[@]}"`. This is a more invasive change that should be reviewed carefully.

### 2. Ambiguous quoting in build_common.sh (SC2140/SC1078/SC1079) — 2 sites

The quoting on cmake flag assignments is technically incorrect, though it works because the variable values contain no spaces.

**`ci/build_common.sh:381,383`**:
```bash
# Current:
echo "The "weighted" time is the elapsed time...
...of how "important" a slow step was...

# Proposed:
echo "The \"weighted\" time is the elapsed time...
...of how \"important\" a slow step was...
```

### 3. Array deduplication with mapfile (SC2207) — 1 site

**`ci/upload_cub_test_artifacts.sh:47`**:
```bash
# Current:
preset_variants=($(echo "${preset_variants[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

# Proposed:
mapfile -t preset_variants < <(printf '%s\n' "${preset_variants[@]}" | sort -u)
```

### 4. Unquoted array assignment (SC2206) — 4 sites

**`ci/build_cub.sh:90-93`**:
```bash
# Current:
CUB_BENCH_FLAGS=($CUB_BENCH_CUB_FLAGS)
CUB_BENCH2_FLAGS=($CUB_BENCH2_CUB_FLAGS)
CUB_BENCH_THRUST_FLAGS=($CUB_BENCH_THRUST_FLAGS_VAR)
CUB_BENCH_FLAGS2=($CUB_BENCH2_FLAGS_VAR)

# Proposed (use read -ra):
read -ra CUB_BENCH_FLAGS <<< "$CUB_BENCH_CUB_FLAGS"
read -ra CUB_BENCH2_FLAGS <<< "$CUB_BENCH2_CUB_FLAGS"
read -ra CUB_BENCH_THRUST_FLAGS <<< "$CUB_BENCH_THRUST_FLAGS_VAR"
read -ra CUB_BENCH_FLAGS2 <<< "$CUB_BENCH2_FLAGS_VAR"
```

### 5. cd without error handling (SC2164) — 1 site

**`ci/test_libcudacxx.sh:3`**:
```bash
# Current:
cd "$(dirname "${BASH_SOURCE[0]}")"

# Proposed:
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
```

### 6. printf instead of echo for escape sequences (SC2028) — 1 site

**`ci/util/build_and_test_targets.sh:89`**:
```bash
# Current:
echo "🔴📝 Configuration override failed ($(elapsed_time)):\n\t${CONFIGURE_OVERRIDE}"

# Proposed:
printf '🔴📝 Configuration override failed (%s):\n\t%s\n' "$(elapsed_time)" "${CONFIGURE_OVERRIDE}"
```

### 7. Unquoted command substitution (SC2046) — 1 site

**`ci/build_cuda_cccl_python.sh:14`**:
```bash
# Current:
echo "Docker socket: " $(ls /var/run/docker.sock)

# Proposed:
echo "Docker socket: $(ls /var/run/docker.sock)"
```

The `ci/pytorch/build_pytorch.sh:123` case (`ninja -C ./build $(xargs -a build/cuda_targets.txt)`) is intentional word splitting — each target must be a separate argument to ninja.

## Detection

Found via ShellCheck 0.11.0 (SC2124, SC2140, SC2207, SC2206, SC2164, SC2028, SC2046). Full analysis: `static-analysis/shellcheck.md`.
