#!/bin/bash

set -euo pipefail

if [ -z "${GITHUB_ACTIONS:-}" ]; then
  echo "This script must be run in a GitHub Actions environment." >&2
  exit 1
fi

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(cd "${ci_dir}/.." && pwd)"

cd "$repo_root"

if ! ci/util/workflow/has_consumers.sh; then
  echo "No consumers found for this job. Exiting." >&2
  exit 0
fi

# Figure out which artifacts need to be built:
consumers=$(ci/util/workflow/get_consumers.sh)
preset_variants=()
if grep -q "TestGPU" <<< "$consumers"; then
  preset_variants+=("no_lid")
fi
if grep -q "HostLaunch" <<< "$consumers"; then
  preset_variants+=("lid_0")
fi
if grep -q "DeviceLaunch" <<< "$consumers"; then
  preset_variants+=("lid_1")
fi
if grep -q "GraphCapture" <<< "$consumers"; then
  preset_variants+=("lid_2")
fi
# Limited jobs run the entire test suite:
if grep -q "SmallGMem" <<< "$consumers"; then
  preset_variants+=("no_lid" "lid_0" "lid_1" "lid_2")
fi

# Remove duplicates:
preset_variants=($(echo "${preset_variants[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

artifact_prefix=z_cub-test-artifacts-$DEVCONTAINER_NAME-${JOB_ID}

# BUILD_INFIX is undefined on windows CI
build_dir_regex="build${CCCL_BUILD_INFIX:+/$CCCL_BUILD_INFIX}/cub[^/]*"

# Just collect the minimum set of files needed for running each ctest preset:
for preset_variant in ${preset_variants[@]}; do

  # Shared across all presets:
  ci/util/artifacts/stage.sh "$artifact_prefix-$preset_variant" \
      "$build_dir_regex/build\.ninja$" \
      "$build_dir_regex/.*rules\.ninja$" \
      "$build_dir_regex/CMakeCache\.txt$" \
      "$build_dir_regex/.*VerifyGlobs\.cmake$" \
      "$build_dir_regex/.*CTestTestfile\.cmake$" \
      > /dev/null

  # Add per-preset executables:
  if [[ "$preset_variant" == lid_* ]]; then
    ci/util/artifacts/stage.sh  \
        "$artifact_prefix-$preset_variant" \
        "$build_dir_regex/bin/.*\.$preset_variant.*" > /dev/null
    # Windows builds generate binaries for the header tests, remove these:
    ci/util/artifacts/unstage.sh  \
        "$artifact_prefix-$preset_variant" \
        "$build_dir_regex/.*\.headers\..*" > /dev/null || :

    ci/util/artifacts/upload_stage_packed.sh "$artifact_prefix-$preset_variant"
  fi
done

if [[ " ${preset_variants[@]} " =~ " no_lid " ]]; then
  # Initially add all binaries to no_lid, then remove the lid variants in later passes:
  ci/util/artifacts/stage.sh \
      "$artifact_prefix-no_lid" \
      "$build_dir_regex/bin/.*" > /dev/null
  # Remove the benchmarks if present, we don't run those in CI, just build them:
  ci/util/artifacts/unstage.sh \
      "$artifact_prefix-no_lid" \
      "$build_dir_regex/.*\.bench\..*" > /dev/null || :
  # Remove all lid variants:
  ci/util/artifacts/unstage.sh \
      "$artifact_prefix-no_lid" \
      "$build_dir_regex/.*\.lid_[0-2].*" > /dev/null
  # Windows builds generate binaries for the header tests, remove these:
  ci/util/artifacts/unstage.sh  \
      "$artifact_prefix-no_lid" \
      "$build_dir_regex/.*\.headers\..*" > /dev/null || :

  # These cubin outputs are needed for FileCheck tests in test/ptx-json
  ci/util/artifacts/stage.sh \
      "$artifact_prefix-no_lid" \
      "$build_dir_regex/cub/test/ptx-json/.*\.cubin$" > /dev/null

  ci/util/artifacts/upload_stage_packed.sh "$artifact_prefix-no_lid"
fi
