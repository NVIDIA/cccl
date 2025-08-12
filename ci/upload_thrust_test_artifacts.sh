#!/bin/bash

set -euo pipefail

if [ -z "${GITHUB_ACTIONS:-}" ]; then
  echo "This script must be run in a GitHub Actions environment." >&2
  exit 1
fi

cd /home/coder/cccl

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

# Just collect the minimum set of files needed for running each ctest preset:
for preset_variant in ${preset_variants[@]}; do
  # Shared across all presets:
  ci/util/artifacts/stage.sh "$artifact_prefix-$preset_variant" \
      "build/${DEVCONTAINER_NAME}/thrust-[^/]+/build\.ninja$" \
      "build/${DEVCONTAINER_NAME}/thrust-[^/]+/.*rules\.ninja$" \
      "build/${DEVCONTAINER_NAME}/thrust-[^/]+/CMakeCache\.txt$" \
      "build/${DEVCONTAINER_NAME}/thrust-[^/]+/.*VerifyGlobs\.cmake$" \
      "build/${DEVCONTAINER_NAME}/thrust-[^/]+/.*CTestTestfile\.cmake$" \
      > /dev/null
done

if [[ " ${preset_variants[@]} " =~ " test_cpu " ]]; then
  # Initially add all binaries, then remove all containing 'cuda' in the name:
  ci/util/artifacts/stage.sh \
      "$artifact_prefix-test_cpu" \
      "build/${DEVCONTAINER_NAME}/thrust-[^/]+/bin/.*" > /dev/null
  ci/util/artifacts/unstage.sh \
      "$artifact_prefix-test_cpu" \
      "build/${DEVCONTAINER_NAME}/thrust-[^/]+/bin/thrust\..*\.cuda\..*" > /dev/null

  ci/util/artifacts/stage.sh \
      "$artifact_prefix-test_cpu" \
      "build/${DEVCONTAINER_NAME}/thrust-[^/]+/lib/libthrust.*" > /dev/null
  ci/util/artifacts/unstage.sh \
      "$artifact_prefix-test_cpu" \
      "build/${DEVCONTAINER_NAME}/thrust-[^/]+/lib/libthrust.*\.cuda\..*" > /dev/null

  ci/util/artifacts/upload_stage_packed.sh "$artifact_prefix-test_cpu"
fi

if [[ " ${preset_variants[@]} " =~ " test_gpu " ]]; then
  # Only binaries containing 'cuda':
  ci/util/artifacts/stage.sh \
      "$artifact_prefix-test_gpu" \
      "build/${DEVCONTAINER_NAME}/thrust-[^/]+/bin/thrust\..*\.cuda\..*" > /dev/null
  ci/util/artifacts/stage.sh \
      "$artifact_prefix-test_gpu" \
      "build/${DEVCONTAINER_NAME}/thrust-[^/]+/lib/libthrust.*\.cuda\..*" > /dev/null

  ci/util/artifacts/upload_stage_packed.sh "$artifact_prefix-test_gpu"
fi
