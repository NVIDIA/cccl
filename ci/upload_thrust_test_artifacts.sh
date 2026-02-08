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

# Figure out which artifact needs to be built:
consumers=$(ci/util/workflow/get_consumers.sh)
preset_variants=()
if grep -q "TestGPU" <<< "$consumers"; then
  preset_variants+=("test_gpu")
fi
if grep -q "TestCPU" <<< "$consumers"; then
  preset_variants+=("test_cpu")
fi

artifact_prefix=z_thrust-test-artifacts-$DEVCONTAINER_NAME-${JOB_ID}

# BUILD_INFIX is undefined on windows CI
build_dir_regex="build${CCCL_BUILD_INFIX:+/$CCCL_BUILD_INFIX}/thrust[^/]*"

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
done

if [[ " ${preset_variants[@]} " =~ " test_cpu " ]]; then
  # Initially add all binaries, then remove all containing 'cuda' in the name:
  ci/util/artifacts/stage.sh \
      "$artifact_prefix-test_cpu" \
      "$build_dir_regex/bin/.*" > /dev/null
  ci/util/artifacts/unstage.sh \
      "$artifact_prefix-test_cpu" \
      "$build_dir_regex/bin/thrust\..*\.cuda\..*" > /dev/null

  ci/util/artifacts/stage.sh \
      "$artifact_prefix-test_cpu" \
      "$build_dir_regex/lib/.*\.test\.framework\..*" > /dev/null
  ci/util/artifacts/unstage.sh \
      "$artifact_prefix-test_cpu" \
      "$build_dir_regex/lib/.*\.cuda\.test\.framework\..*" > /dev/null

  # Windows builds generate binaries for the header tests, remove these:
  ci/util/artifacts/unstage.sh  \
      "$artifact_prefix-test_cpu" \
      "$build_dir_regex/.*\.headers\..*" > /dev/null || :

  ci/util/artifacts/upload_stage_packed.sh "$artifact_prefix-test_cpu"
fi

if [[ " ${preset_variants[@]} " =~ " test_gpu " ]]; then
  # Only binaries containing 'cuda':
  ci/util/artifacts/stage.sh \
      "$artifact_prefix-test_gpu" \
      "$build_dir_regex/bin/thrust\..*\.cuda\..*" > /dev/null
  ci/util/artifacts/stage.sh \
      "$artifact_prefix-test_gpu" \
      "$build_dir_regex/lib/.*\.cuda\.test\.framework\..*" > /dev/null

  # Windows builds generate binaries for the header tests, remove these:
  ci/util/artifacts/unstage.sh  \
      "$artifact_prefix-test_gpu" \
      "$build_dir_regex/.*\.headers\..*" > /dev/null || :


  ci/util/artifacts/upload_stage_packed.sh "$artifact_prefix-test_gpu"
fi
